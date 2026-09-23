"""Itemized Audit Trail and Human/Adjuster-Readable Cost Breakdown Formatter."""
from typing import Dict, List, Any
from .estimator import CostEstimateResult

class CostBreakdownFormatter:
    """Formats cost estimation results into readable Markdown and structured JSON summaries."""

    @staticmethod
    def to_markdown(estimate: CostEstimateResult, title: str = "Repair Cost Breakdown") -> str:
        """Generates formatted Markdown table suitable for insurance adjusters and inspection certificates."""
        cr = estimate.cost_range
        lines = [
            f"### {title}",
            f"**Estimated Cost Range**: ${cr.min_cost:,.2f} - ${cr.max_cost:,.2f} (Expected: **${cr.expected_cost:,.2f} {cr.currency}**)",
            "",
            "| Category | Description | Quantity | Unit Price | Total |",
            "| :--- | :--- | :---: | :---: | :---: |"
        ]

        for item in estimate.line_items:
            cat_badge = item.category.replace("_", " ").title()
            qty_str = f"{item.quantity:.1f} {item.unit}" if item.unit != "flat" else "-"
            lines.append(f"| {cat_badge} | {item.description} | {qty_str} | ${item.unit_price:,.2f} | ${item.total_price:,.2f} |")

        lines.extend([
            "",
            "**Subtotals Summary**:",
            f"- **Parts**: ${estimate.parts_subtotal:,.2f}",
            f"- **Body Labor**: ${estimate.body_labor_subtotal:,.2f}",
            f"- **Paint Labor**: ${estimate.paint_labor_subtotal:,.2f}",
            f"- **Paint Materials**: ${estimate.paint_materials_subtotal:,.2f}",
            f"- **Shop Supplies**: ${estimate.shop_supplies_subtotal:,.2f}",
            f"- **Sales Tax**: ${estimate.tax_total:,.2f}",
            f"- **Vehicle Class Multiplier**: {estimate.vehicle_multiplier:.2f}x",
            "",
            f"**Rationale & Audit Trail**: {estimate.explanation}"
        ])

        return "\n".join(lines)
