from __future__ import annotations

from typing import Any, Dict, List, Optional


def apply_uncertainty_policy(
    result: Dict[str, Any],
    *,
    line_dist_thresh_px: float,
    uncertainty_margin_px: float = 2.0,
    local_y_err_thresh_px: float = 8.0,
    bbox_proxy_spread_thresh_px: float = 17.0,
    left_ankle_visible: Optional[bool] = None,
    right_ankle_visible: Optional[bool] = None,
    comments: str = "",
) -> Dict[str, Any]:
    """
    Convert borderline or weak-geometry decisions into an explicit `uncertain`.

    The policy is intentionally conservative:
    - keep hard failures (`no_goalkeeper`, `no_line`) as `uncertain`
    - abstain near the decision boundary
    - abstain more readily when near-boundary geometry is combined with
      high local y-error or partial ankle visibility
    """

    out = dict(result)
    raw_decision = result.get("decision")
    raw_reason = result.get("reason")
    min_dist = result.get("min_dist")
    local_y_err = result.get("local_y_err")
    point_name = str(result.get("point_name") or "")
    point_source = str(result.get("point_source") or "")
    all_dists = result.get("all_dists") or {}

    flags: List[str] = []

    out["raw_decision"] = raw_decision
    out["raw_reason"] = raw_reason

    if raw_decision == "uncertain":
        out["policy_decision"] = "uncertain"
        out["policy_reason"] = raw_reason or "already_uncertain"
        out["policy_flags"] = ["already_uncertain"]
        out["decision"] = "uncertain"
        out["reason"] = out["policy_reason"]
        return out

    if min_dist is None:
        out["policy_decision"] = "uncertain"
        out["policy_reason"] = "missing_min_dist"
        out["policy_flags"] = ["missing_min_dist"]
        out["decision"] = "uncertain"
        out["reason"] = out["policy_reason"]
        return out

    if local_y_err is None:
        out["policy_decision"] = "uncertain"
        out["policy_reason"] = "missing_local_y_err"
        out["policy_flags"] = ["missing_local_y_err"]
        out["decision"] = "uncertain"
        out["reason"] = out["policy_reason"]
        return out

    min_dist = float(min_dist)
    local_y_err = float(local_y_err)
    comments_lc = (comments or "").strip().lower()

    near_boundary = abs(min_dist - float(line_dist_thresh_px)) <= float(uncertainty_margin_px)
    high_local_y = local_y_err >= float(local_y_err_thresh_px)
    ankle_occlusion = (left_ankle_visible is False) or (right_ankle_visible is False)
    comment_occlusion = "occluded" in comments_lc
    proxy_spread = None
    if all_dists:
        try:
            numeric_dists = [float(v) for v in all_dists.values()]
            proxy_spread = max(numeric_dists) - min(numeric_dists)
        except (TypeError, ValueError):
            proxy_spread = None
    bbox_corner_proxy = point_source == "bbox" and point_name in {"left_bottom", "right_bottom"}
    high_proxy_spread = proxy_spread is not None and proxy_spread >= float(bbox_proxy_spread_thresh_px)

    if near_boundary:
        flags.append("near_boundary")
    if high_local_y:
        flags.append("high_local_y_err")
    if ankle_occlusion:
        flags.append("ankle_occlusion")
    if comment_occlusion:
        flags.append("comment_occlusion")
    if bbox_corner_proxy:
        flags.append("bbox_corner_proxy")
    if high_proxy_spread:
        flags.append("high_proxy_spread")

    policy_decision = raw_decision
    policy_reason = raw_reason or ""

    if raw_decision == "on_line" and bbox_corner_proxy and high_proxy_spread:
        policy_decision = "uncertain"
        policy_reason = "bbox_proxy_spread"
    elif near_boundary and high_local_y:
        policy_decision = "uncertain"
        policy_reason = "boundary_plus_local_y"
    elif near_boundary and ankle_occlusion:
        policy_decision = "uncertain"
        policy_reason = "boundary_plus_ankle_occlusion"
    elif near_boundary and comment_occlusion:
        policy_decision = "uncertain"
        policy_reason = "boundary_plus_comment_occlusion"
    elif near_boundary:
        policy_decision = "uncertain"
        policy_reason = "boundary_margin"

    out["policy_decision"] = policy_decision
    out["policy_reason"] = policy_reason
    out["policy_flags"] = flags
    out["decision"] = policy_decision
    out["reason"] = policy_reason
    return out
