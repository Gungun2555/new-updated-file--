from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class CompetitorTargetEntries(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    hcp_id: str
    hcp_name: Optional[str] = None
    specialty: Optional[str] = None
    territory: Optional[str] = None
    competitor_product: Optional[str] = None
    conversion_potential: Optional[str] = None
    assigned_rep: Optional[str] = None

    class Config:
        orm_mode = True