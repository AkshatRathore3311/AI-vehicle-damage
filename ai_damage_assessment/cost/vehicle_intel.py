"""Automotive Make, Model & Segment Intelligence Resolver."""
from typing import Tuple

class VehicleIntelligence:
    """Classifies vehicle make and model into realistic Indian collision parts & labor tiers."""

    @staticmethod
    def resolve_tier(make_model: str, selected_type: str = "sedan") -> Tuple[str, float, str]:
        s = (make_model or "").lower().strip()
        
        # 1. Premium / Luxury D-Segment, European & Imported CBU/CKD
        luxury_keywords = [
            "bmw", "mercedes", "benz", "audi", "porsche", "jaguar", "land rover", "range rover",
            "lexus", "volvo", "camry", "fortuner", "vellfire", "superb", "kodiaq", "mustang", "mini cooper",
            "toyota camry", "accord"
        ]
        if any(k in s for k in luxury_keywords) or (selected_type or "").lower() == "luxury":
            m_name = make_model.strip() if make_model.strip() and make_model.lower() != "vehicle" else "Luxury Tier"
            return ("luxury", 4.50, f"Premium / Executive OEM ({m_name.title()})")

        # 2. Mid-Size / C-Segment & Compact SUVs
        midsize_keywords = [
            "creta", "seltos", "scorpio", "xuv", "harrier", "safari", "compass", "city", "verna",
            "slavia", "virtus", "innova", "hector", "curvv", "nexon", "brezza", "venue", "sonet", "tigor", "dzire", "ciaz"
        ]
        if any(k in s for k in midsize_keywords) or (selected_type or "").lower() == "suv":
            m_name = make_model.strip() if make_model.strip() and make_model.lower() != "vehicle" else "Mid-Size / SUV"
            return ("suv", 1.45, f"Mid-Size / SUV OEM ({m_name.title()})")

        # 3. Economy / Mass-Market
        economy_keywords = [
            "swift", "baleno", "wagonr", "alto", "i10", "i20", "tiago", "altroz", "kwid", "celerio", "ignis", "spresso"
        ]
        if any(k in s for k in economy_keywords) or (selected_type or "").lower() == "hatchback":
            m_name = make_model.strip() if make_model.strip() and make_model.lower() != "vehicle" else "Economy Hatchback"
            return ("hatchback", 0.85, f"Mass-Market Domestic ({m_name.title()})")

        # 4. Two-wheelers
        if (selected_type or "").lower() == "bike" or any(k in s for k in ["bike", "motorcycle", "scooter", "activa", "bullet", "royalenfield", "ktm", "pulsar"]):
            m_name = make_model.strip() if make_model.strip() and make_model.lower() != "vehicle" else "Two-Wheeler"
            return ("bike", 0.40, f"Two-Wheeler ({m_name.title()})")

        # Default standard passenger
        m_name = make_model.strip() if make_model.strip() and make_model.lower() != "vehicle" else "Standard Passenger Car"
        return ("sedan", 1.00, f"Standard Passenger ({m_name.title()})")
