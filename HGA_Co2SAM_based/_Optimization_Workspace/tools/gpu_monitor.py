import os
import gc
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
import pynvml
import weakref
from typing import List, Dict, Union, Optional, Any
from threading import Lock

class GPUMonitor:
    """
    Global GPU Memory Audit & Process Tracker.

    This class operates as a singleton and provides non-intrusive GPU memory monitoring
    for multi-model deep learning tasks. It identifies physical-layer memory usage,
    logical-layer entity attribution (model/optimizer), and training-pipeline context
    localization.

    Core Mechanisms:
        1. Physical-layer Audit: Uses NVML to directly query GPU hardware state, bypassing
           PyTorch's memory allocator limitations, and identifies memory usage by all PIDs,
           including the current process, child processes, and external intruder processes.
        2. Logical-layer Attribution: Maintains weak reference (Weakref) collections of
           entities, recursively computing physical memory mappings of parameters, buffers,
           and optimizer states for specific models without interfering with garbage collection.
        3. Context Tracking: A stack-based Scope mechanism that precisely captures the logical
           stage at which an OOM event occurs.

    Architecture Overview:
        [User Code] -> [Scope Manager] -> [Snapshot Engine] -> [NVML / torch.cuda]
                             |                  |
                       [Stack Trace]     [Entity Registry]
                             |                  |
                             +------> [Report Generator] ----> [Formatted String]
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUMonitor, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            # Monitoring cannot proceed in this case; halt and report as per "fail-fast" principle
            raise RuntimeError(f"NVML initialization failed; monitoring module cannot start. Error details: {str(e)}")

        self.gpu_ids = None
        self.handles = None
        self.main_pid = os.getpid()
        self.entities = {}  # Storage format: {name: (weakref_obj, type_tag)}
        self.scope_stack = []
        self._initialized = True

    def set_gpu_ids(self, gpu_ids: List[int]):
        """
        Set the list of GPU IDs to be monitored.

        Args:
            gpu_ids (List[int]): List of GPU hardware indices to monitor.
        """
        self.gpu_ids = gpu_ids
        self.handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids}
        
    def register_entity(self, name: str, obj: Union[nn.Module, Optimizer]):
        """
        Register an algorithmic entity (model or optimizer) in the audit list.

        This method uses weak references to store objects, ensuring that monitored
        objects can still be garbage-collected normally.

        Args:
            name (str): Unique identifier for the entity (e.g., 'Student_Model', 'Adam_Optimizer').
            obj (Union[nn.Module, Optimizer]): The PyTorch entity to monitor.
        """
        self.entities[name] = (weakref.ref(obj), type(obj).__name__)

    def scope(self, name: str):
        """
        Create a logical audit context block.

        Wrapping a code block with this context manager provides precise logical
        localization when an OOM occurs. Supports nesting.

        Args:
            name (str): Logical block name (e.g., 'Backbone_Forward', 'Loss_Calculation').

        Returns:
            _ScopeContext: Internal context management object.
        """
        return _ScopeContext(self, name)

    def get_status_line(self) -> str:
        """
        Generate a low-entropy real-time status line for the current state.

        This text includes GPU utilization, the peak memory watermark for the current
        process between two consecutive calls, and real-time interference from external
        processes.

        Core Mechanisms:
            1. Process Attribution: Reports the highest memory allocation watermark reached
               by the current process since the last call to this method.
            2. Auto-Reset: The method automatically resets PyTorch's internal peak statistics
               before returning, ensuring the next reading only reflects the current period's
               pressure.
            3. External Audit: Real-time PID scanning via NVML eliminates the current process
               and its cache interference, precisely identifying external competitors.

        Returns:
            str: Formatted single-line summary (e.g., [GPU0] Util: 85% | Peak: 4200MB | Others: 1200MB | Scope: Forward).
        """
        reports = []
        assert self.gpu_ids, "[Error] ❌ GPU ID list not set! Please call GPU_MONITOR.set_gpu_ids() first."
        for gid in self.gpu_ids:
            assert self.handles
            handle = self.handles[gid]
            # Map physical ID (e.g., 4) to PyTorch's relative logical ID (e.g., 0)
            logical_id = self.gpu_ids.index(gid)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Compute memory occupied by external processes
            self_curr = torch.cuda.memory_allocated(logical_id)
            self_peak = torch.cuda.max_memory_allocated(logical_id)
            others_curr = self._get_external_process_mem(handle)
            
            curr_scope = self.scope_stack[-1] if self.scope_stack else "Global"
            reports.append(
                f"[GPU{gid}] Util: {util_info.gpu}% | "
                f"Peak: {self_peak // 1024**2}MB | "
                f"Others: {others_curr  // 1024**2}MB | "
                f"Scope: {curr_scope}"
            )

            torch.cuda.reset_peak_memory_stats(logical_id)

        return " || ".join(reports)

    def is_oom_exception(self, e: Exception) -> bool:
        """
        Determine whether the given exception is a CUDA out-of-memory error.

        Args:
            e (Exception): The caught exception object.

        Returns:
            bool: True if it is an OOM exception.
        """
        return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()

    def generate_autopsy_report(self) -> str:
        """
        Generate a deep-dive analysis report of the OOM disaster scene (autopsy report).

        This report integrates hardware status, external process competition, internal
        entity distribution, and memory fragmentation.

        Returns:
            str: Multi-line structured diagnostic report.
        """
        report = ["\n" + "="*30 + " GPU OOM AUTOPSY REPORT " + "="*30]
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Logic Path: {' -> '.join(self.scope_stack) if self.scope_stack else 'Global'}")

        assert self.gpu_ids, "[Error] ❌ GPU ID list not set! Please call GPU_MONITOR.set_gpu_ids() first."
        for gid in self.gpu_ids:
            assert self.handles
            handle = self.handles[gid]
            # Map physical ID (e.g., 4) to PyTorch's relative logical ID (e.g., 0)
            logical_id = self.gpu_ids.index(gid)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            report.append(f"\n[Hardware State - GPU{gid}]")
            report.append(f"  - Total: {int(mem_info.total) // 1024**2} MB")
            report.append(f"  - Used: {int(mem_info.used) // 1024**2} MB")
            report.append(f"  - Free: {int(mem_info.free) // 1024**2} MB")

            # External processes
            ext_procs = self._get_detailed_external_processes(handle)
            if ext_procs:
                report.append(f"  - External Competitors (Potential Culprits):")
                for p in ext_procs:
                    report.append(f"    * PID {p['pid']} ({p['name']}): {p['mem'] // 1024**2} MB")
            else:
                report.append(f"  - External Competitors: None")

            # PyTorch memory manager state
            report.append(f"  - PyTorch Internal State (Logic ID {logical_id}):")
            allocated = torch.cuda.memory_allocated(logical_id)
            reserved = torch.cuda.memory_reserved(logical_id)
            report.append(f"  - Allocated: {allocated // 1024**2} MB")
            report.append(f"  - Reserved: {reserved // 1024**2} MB")
            report.append(f"  - Fragmentation: {(reserved - allocated) // 1024**2} MB")

        # Logical entity attribution
        report.append("\n[Logic Entity Attribution]")
        # 1. Collect address-mapping snapshots for all entities
        snapshots = {}
        for name, (ref, _) in self.entities.items():
            obj = ref()
            if obj is not None:
                snapshots[name] = self._calculate_entity_mem(obj)

        # 2. Print individual occupancy audit
        for name, mapping in snapshots.items():
            ind_size = sum(mapping.values())
            report.append(f"  - {name}: {ind_size // 1024**2} MB")

        # 3. Print memory sharing audit
        report.append("\n[Memory Sharing Audit]")
        names = list(snapshots.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                set_i = set(snapshots[names[i]].keys())
                set_j = set(snapshots[names[j]].keys())
                overlap_ptrs = set_i & set_j
                if overlap_ptrs:
                    # Extract the byte count for the overlapping portion using set_i as reference
                    overlap_bytes = sum(snapshots[names[i]][ptr] for ptr in overlap_ptrs)
                    report.append(f"  - Shared ({names[i]} & {names[j]}): {overlap_bytes // 1024**2} MB")

        # 4. Compute physical union (Net Physical Total)
        global_ptrs_map = {}
        for mapping in snapshots.values():
            global_ptrs_map.update(mapping)
        net_physical_bytes = sum(global_ptrs_map.values())
        report.append(f"  - Net Physical Self: {net_physical_bytes // 1024**2} MB")


        # Scan orphan tensors (Top-5)
        # Use the keys from the global union as the set of known addresses
        orphans = self._find_orphan_tensors(set(global_ptrs_map.keys()))
        report.append("\n[Top-5 Orphan Tensors (Unnamed)]")
        for i, (shape, size) in enumerate(orphans[:5]):
            report.append(f"  - Rank {i+1}: Shape {shape} | Size {size // 1024**2} MB")

        report.append("="*84 + "\n")
        return "\n".join(report)

    def _get_external_process_mem(self, handle) -> int:
        """Compute the total GPU memory occupied by processes other than the current process and its children."""
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        total_other = 0
        for p in procs:
            if p.pid != self.main_pid:
                total_other += p.usedGpuMemory
        return total_other

    def _get_detailed_external_processes(self, handle) -> List[Dict]:
        """Retrieve detailed information about external processes."""
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        results = []
        for p in procs:
            if p.pid == self.main_pid:
                continue
            try:
                name = pynvml.nvmlSystemGetProcessName(p.pid).decode('utf-8')
            except:
                name = "Unknown"
            results.append({'pid': p.pid, 'name': name, 'mem': p.usedGpuMemory})
        return results

    def _calculate_entity_mem(self, obj: Any, global_recorded_ptrs: Optional[set] = None) -> Dict[int, int]:
        """
        Compute the physical GPU memory footprint of a specified entity, with support
        for global address deduplication.

        This method uses a recursive discovery engine to search through all nn.Module
        and Optimizer components inside the object, extracting the physical memory
        addresses (data_ptr) of their underlying tensors. It can identify and handle
        parameter sharing across multiple models.

        Core Mechanisms:
            1. Physical Audit: Uses data_ptr to determine whether tensors point to the same
               physical GPU memory block, rather than relying on Python variable names.
            2. Recursive Discovery: Penetrates wrapper classes (e.g., DINO) via _discover_tensors
               to locate deeply nested model components.

        Args:
            obj (Any): The object to audit (e.g., Model, Optimizer, or DINO wrapper class).
            global_recorded_ptrs (set, optional): A set of previously recorded addresses for
                global deduplication. If provided, newly discovered addresses will be
                synchronized into this set.

        Returns:
            Dict[int, int]: Memory address mapping for the entity.
                - Key (int): Physical memory start address (data_ptr).
                - Value (int): Number of bytes occupied by that tensor (byte_size).
                Note: This dictionary is already deduplicated within the entity.
        """
        # 1. Recursively discover all unique tensors held by this entity, {ptr: size}
        discovered_tensors = self._discover_tensors(obj)
        
        entity_ptr_set = set(discovered_tensors.keys())
        entity_total_bytes = sum(discovered_tensors.values())

        # 2. If a global set is provided, synchronize (used to compute the physical union of all algorithm logic)
        if global_recorded_ptrs is not None:
            global_recorded_ptrs.update(entity_ptr_set)

        return self._discover_tensors(obj)

    def _discover_tensors(self, obj: Any, visited_ids: Optional[set] = None) -> Dict[int, int]:
        """
        [Discovery Engine] Recursively search for all PyTorch tensors contained in an object.

        This engine uses reflection to probe object attributes, container members, and
        model parameters. It has native penetration capability for wrapper classes (e.g.,
        DINO Model).

        Search Strategy:
            1. If the object is a Tensor: extract address and size.
            2. If the object is an nn.Module: extract parameters and buffers.
            3. If the object is an Optimizer: extract param_groups and state.
            4. If the object is a list/tuple/dict: recursively traverse members.
            5. If the object is a class instance: recursively traverse public attributes in __dict__.

        Args:
            obj (Any): The target to scan.
            visited_ids (set): Set of object ids already visited, used to break circular references.

        Returns:
            Dict[int, int]: Mapping of discovered tensors {ptr: byte_size}.
        """
        if visited_ids is None:
            visited_ids = set()
        
        obj_id = id(obj)
        if obj_id in visited_ids:
            return {}
        visited_ids.add(obj_id)

        discovered = {}

        # Case A: Directly a tensor
        if torch.is_tensor(obj):
            if obj.is_cuda:
                discovered[obj.data_ptr()] = obj.nelement() * obj.element_size()
            return discovered

        # Case B: PyTorch model component
        if isinstance(obj, nn.Module):
            for p in obj.parameters():
                if p.is_cuda:
                    discovered[p.data_ptr()] = p.nelement() * p.element_size()
                if p.grad is not None and p.grad.is_cuda:
                    discovered[p.grad.data_ptr()] = p.grad.nelement() * p.grad.element_size()
            for b in obj.buffers():
                if b.is_cuda:
                    discovered[b.data_ptr()] = b.nelement() * b.element_size()
            # Note: Modules may internally hold custom non-parameterized sub-objects; continue probing downward
            
        # Case C: Optimizer
        elif isinstance(obj, Optimizer):
            for group in obj.param_groups:
                for p in group['params']:
                    state = obj.state.get(p, {})
                    for v in state.values():
                        if torch.is_tensor(v) and v.is_cuda:
                            discovered[v.data_ptr()] = v.nelement() * v.element_size()

        # Case D: Container types (list, tuple, dict)
        if isinstance(obj, (list, tuple)):
            for item in obj:
                discovered.update(self._discover_tensors(item, visited_ids))
        elif isinstance(obj, dict):
            for item in obj.values():
                discovered.update(self._discover_tensors(item, visited_ids))
        
        # Case E: Plain class instances (e.g., DINO wrappers)
        # Probe all member variables via __dict__, skipping private variables starting with _
        elif hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if not key.startswith('_'):
                    discovered.update(self._discover_tensors(value, visited_ids))

        return discovered

    def _find_orphan_tensors(self, known_ptrs: set) -> List[tuple]:
        """
        Find large tensors that are not in the registered list via GC scanning.
        """
        orphans = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    ptr = obj.data_ptr()
                    if ptr not in known_ptrs:
                        size = obj.nelement() * obj.element_size()
                        if size > 1024**2:  # Only record tensors larger than 1MB
                            orphans.append((list(obj.shape), size))
            except:
                continue
        return sorted(orphans, key=lambda x: x[1], reverse=True)


class _ScopeContext:
    """
    Internal Scope context helper class.
    """
    def __init__(self, monitor: GPUMonitor, name: str):
        self.monitor = monitor
        self.name = name

    def __enter__(self):
        self.monitor.scope_stack.append(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor.scope_stack:
            self.monitor.scope_stack.pop()


# Singleton instantiation
GPU_MONITOR = GPUMonitor()  # Defaults to monitoring GPU 0; modify as needed