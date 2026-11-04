from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel
from src.api.auth import oauth2_scheme, SECRET_KEY, ALGORITHM

class TokenData(BaseModel):
    student_id: str
    tenant_id: str

async def get_current_user_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Dependency that decodes the JWT and validates the signature.
    Returns TokenData containing the authenticated student_id and tenant_id.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        student_id: str = payload.get("student_id")
        tenant_id: str = payload.get("tenant_id")
        
        if student_id is None or tenant_id is None:
            raise credentials_exception
            
        return TokenData(student_id=student_id, tenant_id=tenant_id)
    except JWTError:
        raise credentials_exception
