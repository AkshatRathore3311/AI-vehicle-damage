"""Comprehensive Inspection Report Generator (JSON, HTML, Markdown)."""
import json
from typing import Dict, Any, List
from ..fusion.multi_view_fusion import MultiViewInspectionResult
from ..cost.breakdown import CostBreakdownFormatter

class InspectionReportBuilder:
    """Assembles comprehensive vehicle inspection reports for claims adjusters and automated policy settlement."""

    @staticmethod
    def build_markdown_report(result: MultiViewInspectionResult, vehicle_info: Dict[str, Any]) -> str:
        """Generates human-readable Markdown inspection certificate."""
        cr = result.overall_cost_range
        flag_status = "⚠️ **FLAGGED FOR HUMAN ADJUSTER REVIEW**" if result.requires_human_review else "✅ **AUTOMATED SETTLEMENT APPROVED**"
        
        md_lines = [
            f"# Vehicle Damage Inspection & Repair Cost Assessment",
            f"**Inspection Status**: {flag_status}",
            f"**Vehicle Type / Model**: {vehicle_info.get('make_model', 'Sedan')} ({vehicle_info.get('vehicle_type', 'sedan').upper()})",
            f"**Overall AI Confidence**: {result.overall_confidence*100:.1f}%",
            "",
            "---",
            "## 1. Executive Cost Summary",
            f"- **Estimated Total Repair Cost**: **${cr.expected_cost:,.2f} {cr.currency}**",
            f"- **Confidence Cost Interval**: ${cr.min_cost:,.2f} — ${cr.max_cost:,.2f}",
            f"- **Parts Subtotal**: ${result.overall_parts_cost:,.2f}",
            f"- **Labor Subtotal**: ${result.overall_labor_cost:,.2f}",
            f"- **Paint & Materials Subtotal**: ${result.overall_paint_cost:,.2f}",
            "",
            "---",
            "## 2. Multi-View Damage Deduplication Summary",
            f"- **Total Vehicle Images Inspected**: {result.total_images_processed}",
            f"- **Raw Multi-Angle Detections**: {result.raw_detections_count}",
            f"- **Unique Physical Damage Instances**: {result.fused_unique_damages_count}",
            f"- **Duplicate Detections Eliminated (Deduplicated)**: {result.duplicates_eliminated_count}",
            "",
            "---",
            "## 3. Itemized Damage & Repair Schedule",
            "| ID | Damage Type | Body Part | Severity | Action | Conf | Est. Cost | Observed Views |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
        ]

        for dmg in result.fused_damages:
            views_str = f"Images {dmg.observed_image_indices}"
            md_lines.append(
                f"| {dmg.fused_id} | {dmg.class_name.upper()} | {dmg.part_location.replace('_', ' ').title()} | "
                f"{dmg.severity.level.value.upper()} | {dmg.decision.action.value.upper()} | "
                f"{dmg.best_confidence*100:.0f}% | ${dmg.cost_estimate.cost_range.expected_cost:,.2f} | {views_str} |"
            )

        md_lines.extend([
            "",
            "---",
            "## 4. Engineering Justifications & Audit Trail"
        ])

        for dmg in result.fused_damages:
            md_lines.extend([
                f"### {dmg.fused_id}: {dmg.class_name.title()} on {dmg.part_location.replace('_', ' ').title()}",
                f"- **Recommended Technique**: {dmg.decision.technique}",
                f"- **Structural Rationale**: {dmg.decision.rationale}",
                f"- **Safety Risk**: {dmg.decision.safety_risk}",
                f"- **Cost Explanation**: {dmg.cost_estimate.explanation}",
                ""
            ])

        return "\n".join(md_lines)
