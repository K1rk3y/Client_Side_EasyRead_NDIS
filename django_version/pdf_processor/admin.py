from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, Document, ProcessingJob, ProcessingProgress, GeneratedImage

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ['username', 'email', 'phone', 'company', 'is_staff']
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('phone', 'company')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ('phone', 'company')}),
    )

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Document)
admin.site.register(ProcessingJob)
admin.site.register(ProcessingProgress)
admin.site.register(GeneratedImage)