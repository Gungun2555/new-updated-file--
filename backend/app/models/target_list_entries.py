from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class TargetListEntriesModel(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    hcp_id: str
    hcp_name: Optional[str] = None
    specialty: Optional[str] = None
    territory: Optional[str] = None
    tier: Optional[str] = None