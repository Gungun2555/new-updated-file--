from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class CallListEntriesModel(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    hcp_id: str
    hcp_name: Optional[str] = None
    call_date: date
    sales_rep: Optional[str] = None
    status: Optional[str] = None