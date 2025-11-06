from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class ListVersions(BaseModel):
    version_id: Optional[int] = None
    request_id: int
    version_number: int
    change_type: str
    change_rationale: str
    created_by: str
    is_current: Optional[bool] = None
    created_at: Optional[datetime] = None

    class Config:
        orm_mode = True