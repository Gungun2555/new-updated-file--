from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class WorkLogsModel(BaseModel):
    log_id: Optional[int] = None
    request_id: int
    version_id: Optional[int] = None
    worker_name: str
    activity_description: str
    decisions_made: Optional[str] = None
    activity_date: Optional[datetime] = None