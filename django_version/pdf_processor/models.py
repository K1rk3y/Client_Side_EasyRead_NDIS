from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

# 自定义用户模型
class CustomUser(AbstractUser):
    # 添加自定义字段
    phone = models.CharField(max_length=20, blank=True)
    company = models.CharField(max_length=100, blank=True)
    
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to.',
        related_name='custom_users',
        related_query_name='custom_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name='custom_users',
        related_query_name='custom_user',
    )

    def __str__(self):
        return f"{self.username} ({self.company})"

class Document(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='documents/')
    uploaded_by = models.ForeignKey('CustomUser', on_delete=models.CASCADE)  # 使用自定义用户
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

class ProcessingJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Job for {self.document.title} - {self.status}"

class ProcessingProgress(models.Model):
    user = models.ForeignKey('CustomUser', on_delete=models.CASCADE)  # 使用自定义用户
    job = models.ForeignKey(ProcessingJob, on_delete=models.CASCADE)
    progress_percentage = models.IntegerField(default=0)
    current_step = models.CharField(max_length=100, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.progress_percentage}%"

class GeneratedImage(models.Model):
    job = models.ForeignKey(ProcessingJob, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='generated_images/')
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image for {self.job.document.title}"