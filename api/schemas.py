from pydantic import BaseModel

class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    DAYS_BIRTH: int