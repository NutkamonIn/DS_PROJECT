import re

def preprocess_text(text: str) -> str:
    """
    ฟังก์ชันสำหรับทำความสะอาดข้อความ (Text Preprocessing)
    ตามข้อกำหนดของ Assignment:
    1. Whitespace Normalization (ลดช่องว่างซ้ำ)
    2. Lowercase (สำหรับภาษาอังกฤษ)
    3. Basic Normalize (ตัดช่องว่างหัวท้าย)
    
    *ข้อควรระวัง: ห้ามทำ Over-cleaning เช่น ลบ Emoji หรือ Slang*
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Basic Normalize: ตัดช่องว่างหัวท้าย
    text = text.strip()
    
    # 2. Whitespace Normalization: เปลี่ยนช่องว่างหลายๆ ตัว รวมถึง tab/newline ให้เป็น space เดียว
    # ตัวอย่าง: "สวัสดี    ครับ" -> "สวัสดี ครับ"
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Lowercase: แปลงตัวอักษรภาษาอังกฤษเป็นตัวพิมพ์เล็ก (เพื่อให้ case-insensitive)
    # ตัวอย่าง: "iPhone รุ่นใหม่" -> "iphone รุ่นใหม่"
    text = text.lower()
    
    return text
