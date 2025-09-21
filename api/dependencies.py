"""
Общие зависимости для API endpoints
"""

from typing import Optional
from fastapi import Depends, HTTPException, status

async def get_current_user():
    """Получение текущего пользователя (заглушка)"""
    # Здесь будет логика аутентификации
    return {"user_id": "anonymous", "role": "user"}

async def require_admin():
    """Требование прав администратора"""
    user = await get_current_user()
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin rights required"
        )
    return user

def get_pagination(skip: int = 0, limit: int = 100):
    """Параметры пагинации"""
    if limit > 1000:
        limit = 1000
    return {"skip": skip, "limit": limit}