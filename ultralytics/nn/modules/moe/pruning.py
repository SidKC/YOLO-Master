# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""Pruning utilities for Mixture-of-Experts models"""
import torch
import torch.nn as nn
import copy
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .analysis import ExpertUsageTracker


class MoEPruner:
    """Pruner for Mixture-of-Experts models based on usage statistics"""
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.15,
        dataset: str = 'coco8.yaml',
        device: Optional[str] = None,
        importance_mode: str = "usage",
        keep_top_m: Optional[int] = None,
        eval_dataset: Optional[str] = None,
        moe_inference_mode: str = "dense",
        signal_json: Optional[str] = None,
    ):
        """
        Initialize MoE pruner
        
        Args:
            model_path: Path to the model file
            threshold: Minimum usage percentage to keep an expert (0.0-1.0)
            dataset: Dataset configuration for validation
            device: Device for validation. ``None`` (default) auto-detects CUDA,
                falling back to CPU — previously hard-coded to 'cpu', which was
                needlessly slow on GPU boxes.
            importance_mode: ``usage`` for hard hit frequency, ``usage_weight``
                for hit frequency times average gate weight, ``avg_weight`` for
                gate weight alone, or ``soft_contribution`` for normalized soft
                contribution mass within each layer.
            keep_top_m: Optional fixed expert budget used for diagnostic ablations.
        """
        valid_modes = {"usage", "usage_weight", "avg_weight", "soft_contribution"}
        if importance_mode not in valid_modes:
            raise ValueError(f"importance_mode must be one of {sorted(valid_modes)}, got {importance_mode!r}")
        if keep_top_m is not None and keep_top_m < 1:
            raise ValueError("keep_top_m must be >= 1 when provided")

        self.model_path = model_path
        self.threshold = threshold
        self.dataset = dataset
        self.device = device if device is not None else self._auto_device()
        self.importance_mode = importance_mode
        self.keep_top_m = keep_top_m
        self.eval_dataset = eval_dataset or dataset
        if moe_inference_mode not in {"dense", "sparse"}:
            raise ValueError("moe_inference_mode must be 'dense' or 'sparse'")
        self.moe_inference_mode = moe_inference_mode
        self.signal_json = signal_json
        self.signal_sha256 = None
        self.model = None
        self.usage_stats: Dict[str, Dict[int, Any]] = {}
        self.pruning_plan: Dict[str, List[int]] = {}

    @staticmethod
    def _auto_device() -> str:
        """Pick CUDA when available, else MPS, else CPU."""
        import torch
        if torch.cuda.is_available():
            return '0'
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
        
    def _load_model(self) -> None:
        """Load YOLO model from file"""
        from ultralytics import YOLO
        
        try:
            self.model = YOLO(self.model_path)
            for module in self.model.model.modules():
                ensure_compat = getattr(module, "_ensure_compat_attrs", None)
                if callable(ensure_compat):
                    ensure_compat()
                if hasattr(module, "use_sparse_inference"):
                    module.use_sparse_inference = self.moe_inference_mode == "sparse"
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _diagnose_usage(self) -> None:
        """Run diagnosis to collect expert usage statistics"""
        print("\n[Phase 1] Diagnosing Expert Usage...")
        
        with ExpertUsageTracker(self.model.model) as tracker:
            try:
                self.model.val(
                    data=self.dataset, 
                    split='val', 
                    batch=1, 
                    verbose=False, 
                    device=self.device
                )
                self.usage_stats = tracker.usage_stats
                print(f"✅ Collected usage stats for {len(self.usage_stats)} layers")
            except Exception as e:
                raise RuntimeError(f"Diagnosis failed: {e}")

    def _expert_score(self, expert_stats: Any, total_hits: float) -> float:
        """Return the raw expert importance score for the configured signal."""
        usage_pct = float(expert_stats.hits) / total_hits if total_hits > 0 else 0.0
        avg_weight = float(getattr(expert_stats, "avg_weight", 0.0))
        if self.importance_mode in {"usage_weight", "soft_contribution"}:
            return usage_pct * avg_weight
        if self.importance_mode == "avg_weight":
            return avg_weight
        return usage_pct
    
    def _create_pruning_plan(self) -> None:
        """Create pruning plan based on usage statistics"""
        print("\n[Phase 2] Planning Surgery...")
        
        modules_dict = dict(self.model.model.named_modules())
        
        for layer_name, stats in self.usage_stats.items():
            total_hits = sum(s.hits for s in stats.values())
            if total_hits == 0:
                continue
            
            expert_scores = {
                expert_id: self._expert_score(expert_stats, total_hits)
                for expert_id, expert_stats in stats.items()
            }
            if self.importance_mode == "soft_contribution":
                score_sum = sum(expert_scores.values())
                if score_sum > 0:
                    expert_scores = {expert_id: score / score_sum for expert_id, score in expert_scores.items()}

            experts_to_keep = []
            print(f"\n   Layer: {layer_name}")

            if self.keep_top_m is not None:
                keep_count = min(self.keep_top_m, len(expert_scores))
                experts_to_keep = [
                    expert_id
                    for expert_id, _ in sorted(expert_scores.items(), key=lambda item: (-item[1], item[0]))[:keep_count]
                ]

            # Determine which experts to keep based on threshold or fixed budget.
            for expert_id, expert_stats in sorted(stats.items()):
                usage_pct = expert_stats.hits / total_hits
                score = expert_scores[expert_id]
                if self.keep_top_m is None and score >= self.threshold:
                    experts_to_keep.append(expert_id)
                if expert_id in experts_to_keep:
                    print(f"     ✅ Keep E{expert_id} (Usage: {usage_pct:.1%}, Score: {score:.4f})")
                else:
                    print(f"     🗑️  Drop E{expert_id} (Usage: {usage_pct:.1%}, Score: {score:.4f})")
            
            # Safety check: ensure at least one expert remains
            if len(experts_to_keep) == 0:
                print(f"     ❌ Error: All experts would be pruned! Keeping top expert.")
                top_expert = max(expert_scores.items(), key=lambda item: item[1])[0]
                experts_to_keep = [top_expert]
            
            # Check against original top_k requirement
            if layer_name in modules_dict:
                module = modules_dict[layer_name]
                original_top_k = getattr(module, 'top_k', 2)
                
                if len(experts_to_keep) < original_top_k:
                    print(f"     ⚠️  Warning: Keeping {len(experts_to_keep)} experts, "
                          f"but original top_k={original_top_k}")
            
            self.pruning_plan[layer_name] = sorted(experts_to_keep)
        
        print(f"\n✅ Pruning plan created for {len(self.pruning_plan)} layers")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_pruning_plan_from_signal_json(self) -> None:
        """Build a pruning plan from one immutable calibration signal artifact."""
        signal_path = Path(self.signal_json)
        payload = json.loads(signal_path.read_text(encoding="utf-8"))
        expected_model_hash = payload.get("model_sha256")
        actual_model_hash = self._sha256_file(Path(self.model_path))
        if expected_model_hash and expected_model_hash != actual_model_hash:
            raise RuntimeError(
                f"Signal/model hash mismatch: signal={expected_model_hash}, model={actual_model_hash}"
            )

        modules_dict = dict(self.model.model.named_modules())
        self.pruning_plan = {}
        for layer_name, layer in sorted(payload.get("layers", {}).items()):
            if layer_name not in modules_dict:
                raise RuntimeError(f"Signal layer not found in model: {layer_name}")
            scores = {}
            for expert in layer.get("experts", []):
                expert_id = int(expert["expert_id"])
                if self.importance_mode == "soft_contribution":
                    score = float(expert.get("soft_contribution", 0.0))
                elif self.importance_mode == "avg_weight":
                    score = float(expert.get("average_gate_weight", 0.0))
                elif self.importance_mode == "usage_weight":
                    score = float(expert.get("hard_usage", 0.0)) * float(
                        expert.get("average_gate_weight", 0.0)
                    )
                else:
                    score = float(expert.get("hard_usage", 0.0))
                scores[expert_id] = score
            if not scores:
                raise RuntimeError(f"Signal layer has no expert scores: {layer_name}")

            ranked = sorted(scores, key=lambda expert_id: (-scores[expert_id], expert_id))
            if self.keep_top_m is not None:
                keep = ranked[: min(self.keep_top_m, len(ranked))]
            else:
                keep = [expert_id for expert_id in sorted(scores) if scores[expert_id] >= self.threshold]
            if not keep:
                keep = ranked[:1]
            self.pruning_plan[layer_name] = sorted(keep)
            print(
                f"   Signal plan {layer_name}: keep={sorted(keep)} "
                f"scores={{{', '.join(f'{key}: {scores[key]:.6f}' for key in sorted(scores))}}}"
            )

        if not self.pruning_plan:
            raise RuntimeError(f"Signal JSON contains no usable layers: {signal_path}")
        self.signal_sha256 = self._sha256_file(signal_path)
        print(f"✅ Loaded immutable pruning plan from {signal_path}")
    
    def _get_parent_module_name(self, layer_name: str) -> str:
        """
        Extract parent module name from layer name
        
        Args:
            layer_name: Full layer name (e.g., 'model.x.routing')
            
        Returns:
            Parent module name (e.g., 'model.x')
        """
        parts = layer_name.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else ""
    
    def _find_projection_layer(
        self, 
        router: nn.Module, 
        num_experts: int
    ) -> Optional[Tuple[nn.Module, str]]:
        """
        Find the projection layer in router that outputs to experts
        
        Args:
            router: Router module
            num_experts: Original number of experts
            
        Returns:
            Tuple of (projection_layer, layer_path) or None if not found
        """
        # Check common router structures
        candidates = [
            ('router', 'router'),
            ('routing_network', 'routing_network'),
        ]
        
        for attr_name, path_name in candidates:
            if hasattr(router, attr_name):
                sequential = getattr(router, attr_name)
                if isinstance(sequential, nn.Sequential) and len(sequential) > 0:
                    last_layer = sequential[-1]
                    
                    # Check if it's the projection layer
                    if isinstance(last_layer, nn.Conv2d):
                        if last_layer.out_channels == num_experts:
                            return last_layer, f"{path_name}[-1]"
                    elif isinstance(last_layer, nn.Linear):
                        if last_layer.out_features == num_experts:
                            return last_layer, f"{path_name}[-1]"
        
        return None
    
    def _prune_experts(
        self, 
        moe_module: nn.Module, 
        keep_indices: List[int]
    ) -> None:
        """
        Prune expert modules
        
        Args:
            moe_module: MoE module containing experts
            keep_indices: Indices of experts to keep
        """
        old_experts = moe_module.experts
        new_experts = nn.ModuleList([old_experts[i] for i in keep_indices])
        
        moe_module.experts = new_experts
        moe_module.num_experts = len(keep_indices)
        if hasattr(moe_module, "expert_usage_counts"):
            previous = moe_module.expert_usage_counts
            moe_module.expert_usage_counts = previous.new_zeros(len(keep_indices))
        if hasattr(moe_module, "last_routing_snapshot"):
            moe_module.last_routing_snapshot = {}
        
        # Adjust top_k if necessary
        if hasattr(moe_module, 'top_k') and moe_module.top_k > moe_module.num_experts:
            old_top_k = moe_module.top_k
            moe_module.top_k = moe_module.num_experts
            print(f"     📉 Reduced top_k from {old_top_k} to {moe_module.top_k}")
    
    def _prune_router_weights(
        self, 
        router: nn.Module, 
        keep_indices: List[int],
        num_old_experts: int
    ) -> bool:
        """
        Prune router projection layer weights
        
        Args:
            router: Router module
            keep_indices: Indices of experts to keep
            num_old_experts: Original number of experts
            
        Returns:
            True if successful, False otherwise
        """
        result = self._find_projection_layer(router, num_old_experts)
        
        if result is None:
            print(f"     ⚠️  Could not locate router projection layer. "
                  f"Skipping weight pruning.")
            return False
        
        proj_layer, layer_path = result
        print(f"     ✂️  Pruning router projection ({layer_path})")
        
        # Create new projection layer with reduced output dimension
        if isinstance(proj_layer, nn.Conv2d):
            new_proj = nn.Conv2d(
                in_channels=proj_layer.in_channels,
                out_channels=len(keep_indices),
                kernel_size=proj_layer.kernel_size,
                stride=proj_layer.stride,
                padding=proj_layer.padding,
                bias=(proj_layer.bias is not None)
            ).to(device=proj_layer.weight.device, dtype=proj_layer.weight.dtype)
        elif isinstance(proj_layer, nn.Linear):
            new_proj = nn.Linear(
                in_features=proj_layer.in_features,
                out_features=len(keep_indices),
                bias=(proj_layer.bias is not None)
            ).to(device=proj_layer.weight.device, dtype=proj_layer.weight.dtype)
        else:
            return False
        
        # Copy weights for kept experts
        with torch.no_grad():
            new_proj.weight.data = proj_layer.weight.data[keep_indices].clone()
            if proj_layer.bias is not None:
                new_proj.bias.data = proj_layer.bias.data[keep_indices].clone()
        new_proj.weight.requires_grad_(proj_layer.weight.requires_grad)
        if new_proj.bias is not None and proj_layer.bias is not None:
            new_proj.bias.requires_grad_(proj_layer.bias.requires_grad)
        
        # Replace the layer in the sequential container
        if 'routing_network' in layer_path:
            router.routing_network[-1] = new_proj
        elif 'router' in layer_path:
            router.router[-1] = new_proj
        
        # Update router attributes
        router.num_experts = len(keep_indices)
        if hasattr(router, 'top_k'):
            router.top_k = min(router.top_k, router.num_experts)
        new_proj.train(proj_layer.training)
        
        return True

    def _validate_pruned_module(self, moe_module: nn.Module) -> None:
        """Fail fast when expert, router, or Top-K dimensions diverge after surgery."""
        num_experts = int(moe_module.num_experts)
        if len(moe_module.experts) != num_experts:
            raise RuntimeError("expert list length does not match num_experts after pruning")
        if int(getattr(moe_module.routing, "num_experts", -1)) != num_experts:
            raise RuntimeError("router num_experts does not match pruned expert count")
        if int(getattr(moe_module, "top_k", num_experts)) > num_experts:
            raise RuntimeError("top_k exceeds pruned expert count")
        projection = self._find_projection_layer(moe_module.routing, num_experts)
        if projection is None:
            raise RuntimeError("router projection dimension does not match pruned expert count")
    
    def _perform_surgery(self) -> nn.Module:
        """
        Perform actual pruning surgery on the model
        
        Returns:
            Pruned model
        """
        print("\n[Phase 3] Performing Surgery...")
        
        new_model = copy.deepcopy(self.model.model)
        modules_dict = dict(new_model.named_modules())
        
        for layer_name, keep_indices in self.pruning_plan.items():
            # Get parent MoE module
            parent_name = self._get_parent_module_name(layer_name)
            if not parent_name:
                print(f"   ❌ Could not determine parent module for {layer_name}")
                continue
            
            if parent_name not in modules_dict:
                print(f"   ❌ Parent module {parent_name} not found")
                continue
            
            moe_module = modules_dict[parent_name]
            
            # Verify MoE structure
            if not hasattr(moe_module, 'experts') or not hasattr(moe_module, 'routing'):
                print(f"   ❌ {parent_name} missing 'experts' or 'routing' attributes")
                continue
            
            num_old_experts = len(moe_module.experts)
            
            # Skip if no pruning needed
            if len(keep_indices) == num_old_experts:
                print(f"   ⏭️  Skipping {layer_name} (no changes needed)")
                continue
            
            print(f"   🔧 Pruning {layer_name}")
            print(f"     Experts: {num_old_experts} → {len(keep_indices)} "
                  f"(keeping {keep_indices})")
            
            if self._find_projection_layer(moe_module.routing, num_old_experts) is None:
                raise RuntimeError(f"Cannot atomically prune {parent_name}: router projection was not found")

            # Prune experts
            self._prune_experts(moe_module, keep_indices)
            
            # Prune router weights
            router_pruned = self._prune_router_weights(
                moe_module.routing, 
                keep_indices, 
                num_old_experts
            )
            if not router_pruned:
                raise RuntimeError(f"Cannot atomically prune {parent_name}: router pruning failed")
            self._validate_pruned_module(moe_module)
        
        print("\n✅ Surgery completed")
        return new_model
    
    def _save_model(self, pruned_model: nn.Module, output_path: str) -> None:
        """
        Save pruned model to file
        
        Args:
            pruned_model: Pruned model
            output_path: Output file path
        """
        print(f"\n[Phase 4] Saving Pruned Model...")
        
        # Update YOLO wrapper
        self.model.model = pruned_model
        
        # Save checkpoint
        checkpoint = {
            'model': pruned_model,
            'updates': None,
            'pruning_info': {
                'threshold': self.threshold,
                'pruning_plan': self.pruning_plan,
                'importance_mode': self.importance_mode,
                'calibration_dataset': self.dataset,
                'eval_dataset': self.eval_dataset,
                'moe_inference_mode': self.moe_inference_mode,
                'signal_json': self.signal_json,
                'signal_sha256': self.signal_sha256,
            }
        }
        
        torch.save(checkpoint, output_path)
        print(f"✅ Saved to: {output_path}")
    
    def _verify_model(self, output_path: str) -> bool:
        """
        Verify pruned model can be loaded and validated
        
        Args:
            output_path: Path to pruned model
            
        Returns:
            True if verification successful
        """
        print("\n[Phase 5] Verification...")
        
        try:
            from ultralytics import YOLO
            
            # Load check
            pruned_model = YOLO(output_path)
            for module in pruned_model.model.modules():
                if hasattr(module, "use_sparse_inference"):
                    module.use_sparse_inference = self.moe_inference_mode == "sparse"
            print("   ✅ Load check: OK")
            
            # Validation check
            print("   🔄 Running validation on pruned model...")
            pruned_model.val(
                data=self.eval_dataset,
                split='val', 
                batch=1, 
                verbose=False, 
                device=self.device
            )
            print("   ✅ Validation check: OK")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            return False
    
    def prune(self, output_path: str) -> bool:
        """
        Execute complete pruning pipeline
        
        Args:
            output_path: Path to save pruned model
            
        Returns:
            True if pruning successful
        """
        print(f"\n{'='*80}")
        print(f"✂️  MoE MODEL PRUNING PIPELINE".center(80))
        print(f"{'='*80}")
        print(f"\n📋 Configuration:")
        print(f"   • Input Model: {self.model_path}")
        print(f"   • Output Model: {output_path}")
        print(f"   • Usage Threshold: {self.threshold*100:.1f}%")
        print(f"   • Dataset: {self.dataset}")
        
        try:
            # Phase 1: Load model
            self._load_model()
            
            # Phase 2-3: Reuse immutable calibration stats, or diagnose once.
            if self.signal_json:
                self._load_pruning_plan_from_signal_json()
            else:
                self._diagnose_usage()
                self._create_pruning_plan()
            
            # Phase 4: Perform surgery
            pruned_model = self._perform_surgery()
            
            # Phase 5: Save model
            self._save_model(pruned_model, output_path)
            
            # Phase 6: Verify
            success = self._verify_model(output_path)
            
            if success:
                print(f"\n{'='*80}")
                print(f"🎉 PRUNING COMPLETED SUCCESSFULLY".center(80))
                print(f"{'='*80}\n")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Pruning failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def prune_moe_model(
    model_path: str, 
    output_path: str, 
    threshold: float = 0.15, 
    dataset: str = 'coco8.yaml',
    importance_mode: str = "usage",
    keep_top_m: Optional[int] = None,
    device: Optional[str] = None,
    eval_dataset: Optional[str] = None,
    moe_inference_mode: str = "dense",
    signal_json: Optional[str] = None,
) -> bool:
    """
    Prune MoE model by removing underutilized experts
    
    Args:
        model_path: Path to input model file
        output_path: Path to save pruned model
        threshold: Minimum usage percentage to keep expert (0.0-1.0)
        dataset: Dataset configuration for validation
        
    Returns:
        True if pruning successful
    """
    pruner = MoEPruner(
        model_path,
        threshold,
        dataset,
        device=device,
        importance_mode=importance_mode,
        keep_top_m=keep_top_m,
        eval_dataset=eval_dataset,
        moe_inference_mode=moe_inference_mode,
        signal_json=signal_json,
    )
    return pruner.prune(output_path)


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Prune underutilized experts from MoE YOLO models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path", 
        help="Path to input model file (.pt)"
    )
    parser.add_argument(
        "--output", 
        default="pruned_model.pt", 
        help="Path to save pruned model"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.15, 
        help="Minimum usage percentage to keep expert (0.0-1.0)"
    )
    parser.add_argument(
        "--dataset",
        default="coco8.yaml",
        help="Dataset configuration for validation"
    )
    parser.add_argument(
        "--signal-json",
        default=None,
        help="Immutable calibration artifact from diagnose_moe_pruning_signal.py",
    )
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help="Dataset configuration used only for post-surgery validation",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device used for routing diagnosis and verification",
    )
    parser.add_argument(
        "--moe-inference-mode",
        choices=("dense", "sparse"),
        default="dense",
        help="ES_MOE execution path shared by diagnosis and verification",
    )
    parser.add_argument(
        "--importance-mode",
        choices=("usage", "usage_weight", "avg_weight", "soft_contribution"),
        default="usage",
        help="Expert importance signal used by the pruning threshold",
    )
    parser.add_argument(
        "--keep-top-m",
        type=int,
        default=None,
        help="Keep exactly the top-M experts per layer for a fixed-budget ablation",
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    success = prune_moe_model(
        args.model_path, 
        args.output, 
        args.threshold,
        args.dataset,
        importance_mode=args.importance_mode,
        keep_top_m=args.keep_top_m,
        device=args.device,
        eval_dataset=args.eval_dataset,
        moe_inference_mode=args.moe_inference_mode,
        signal_json=args.signal_json,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
