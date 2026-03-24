import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import matplotlib.pyplot as plt

# الحصول على مسار المجلد الحالي للبرنامج
script_dir = os.path.dirname(os.path.abspath(__file__))

# إعداد المجلد للنتائج (داخل مجلد البرنامج)
output_dir = os.path.join(script_dir, "masked images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# اختيار الجهاز (GPU أو CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# تحميل نموذج Mask R-CNN 
print("Loading Mask R-CNN...")
model = maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# تحميل الصورة (مسار نسبي إلى موقع البرنامج)
image_path = os.path.join(script_dir, "image/group walking.jpg")
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    print(f"Image loaded successfully: {image_path}")
    original_shape = image.shape
    print(f"Original image size: {image.shape}")
    
    # تصغير الصورة لتسريع المعالجة (البقاء في نفس النسبة)
    max_size = 800  # الحد الأقصى للعرض والارتفاع
    scale = 1.0
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        new_height = int(image.shape[0] * scale)
        new_width = int(image.shape[1] * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"Resized image size: {image.shape}")
    
    # تحويل الصورة من BGR إلى RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # تحويل الصورة إلى tensor وتطبيع القيم
    img_tensor = F.to_tensor(image_rgb).to(device)
    
    # التنبؤ باستخدام Mask R-CNN
    print("Detecting people...")
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # استخراج النتائج
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()
    
    # فلترة للحصول على الأشخاص فقط (class 1 في COCO)
    person_class_id = 1  # الشخص في COCO dataset
    confidence_threshold = 0.5
    
    person_indices = np.where((labels == person_class_id) & (scores >= confidence_threshold))[0]
    
    print(f"Number of people detected: {len(person_indices)}")
    
    # إنشاء صورة مع masking
    masked_image = image.copy()
    
    if len(person_indices) > 0:
        # دمج جميع أقنعة الأشخاص
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        for idx in person_indices:
            # الحصول على الـ mask (يكون في shape [1, H, W])
            mask = masks[idx, 0] > 0.5
            combined_mask = combined_mask | mask
        
        # تطبيق التلوين على الأقنعة
        # اللون الأحمر (BGR: 0, 0, 255)
        mask_color = np.array([0, 0, 255])
        
        # دمج اللون مع الصورة الأصلية
        masked_image[combined_mask] = masked_image[combined_mask] * 0.3 + mask_color * 0.7
    
    # إعادة تصغير الصورة إلى حجمها الأصلي إذا تم تصغيرها
    if scale < 1.0:
        masked_image = cv2.resize(masked_image, (original_shape[1], original_shape[0]))
    
    # حفظ النتيجة
    output_path = os.path.join(output_dir, "masked_people.jpg")
    cv2.imwrite(output_path, masked_image)
    print(f"✓ Image saved successfully: {output_path}")
    
    # خيار: حفظ الصورة مع رسم الصناديق والأقنعة
    annotated_image = image.copy()
    for idx in person_indices:
        box = boxes[idx].astype(int)
        score = scores[idx]
        
        # رسم الصندوق
        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # كتابة درجة الثقة
        cv2.putText(annotated_image, f'{score:.2f}', (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # رسم الـ mask على الصورة
        mask = masks[idx, 0] > 0.5
        annotated_image[mask] = annotated_image[mask] * 0.6 + np.array([0, 255, 0]) * 0.4
    
    # إعادة تصغير صورة الكشف أيضاً
    if scale < 1.0:
        annotated_image = cv2.resize(annotated_image, (original_shape[1], original_shape[0]))
    
    detection_output_path = os.path.join(output_dir, "detected_people.jpg")
    cv2.imwrite(detection_output_path, annotated_image)
    print(f"✓ Image saved successfully: {detection_output_path}")
    
    # إعادة تصغير الصورة الأصلية أيضاً للمقارنة
    if scale < 1.0:
        original_resized = cv2.resize(image, (original_shape[1], original_shape[0]))
    else:
        original_resized = image.copy()
    
    # تحميل الصور للمقارنة
    original_display = cv2.imread(image_path)
    masked_display = cv2.imread(output_path)
    detected_display = cv2.imread(detection_output_path)
    
    # إنشاء صورة مقارنة بثلاث صور جنب بعض
    comparison = np.hstack([original_display, detected_display, masked_display])
    
    comparison_path = os.path.join(output_dir, "full_comparison.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ Image saved successfully: {comparison_path}")
    
    print("✓ Completed successfully!")
    
    # ========== عرض النتائج با Matplotlib ==========
    print("\n📊 Jaring to display results...")
    
    # تحميل الصور
    original = cv2.imread(image_path)
    detected = cv2.imread(detection_output_path)
    masked = cv2.imread(output_path)
    
    # تحويل من BGR إلى RGB للعرض الصحيح
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    
    # عرض الصور
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # الصورة الأصلية
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold', color='navy')
    axes[0].axis('off')
    
    # صورة الكشف
    axes[1].imshow(detected_rgb)
    axes[1].set_title("Detection with Boxes", fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # صورة Segmentation
    axes[2].imshow(masked_rgb)
    axes[2].set_title("Segmentation Masks", fontsize=14, fontweight='bold', color='red')
    axes[2].axis('off')
    
    plt.suptitle(f"Mask R-CNN Results - {len(person_indices)} People Detected", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("✓ Displayed results successfully!")