from django.db import models
from django.contrib.auth.models import AbstractUser

# 重命名为 CustomUser 或 PdfUser，避免与内置 User 冲突
class PdfUser(models.Model):
    # 如果只是扩展字段，使用 OneToOneField 关联到内置 User
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE, related_name='pdf_profile')
    # 添加你的自定义字段
    phone = models.CharField(max_length=20, blank=True)
    company = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - PDF Profile"

# 或者如果你需要完全自定义用户模型，继承 AbstractUser
class CustomUser(AbstractUser):
    # 添加自定义字段
    phone = models.CharField(max_length=20, blank=True)
    company = models.CharField(max_length=100, blank=True)
    
    # 添加 related_name 来避免冲突
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_name='custom_users'  # 添加这个
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='custom_users'  # 添加这个
    )

class ProcessingProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    progress = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
