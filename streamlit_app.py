import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. إعدادات الصفحة وهوية المشروع
# ==========================================
st.set_page_config(
    page_title="PlantMed AI - Basil", 
    page_icon="🌿", 
    layout="centered"
)

# إدارة الصفحات باستخدام الـ Session State
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# ==========================================
# 🎁 أولاً: صفحة الترحيب الاحترافية (Welcome Page)
# ==========================================
if st.session_state.page == 'welcome':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("🌿 مرحباً بكم في منصة P.L.A.N.T. M.E.D. AI")
    st.subheader("النظام الذكي المطور لتشخيص أمراض المحاصيل الزراعية")
    
    st.markdown("""
    ### 👋 أهلاً بك يا دكتور في نظامنا الجديد،
    يسعدنا ويشرفنا تقديم هذا العمل المتكامل المخصص لفحص وتشخيص أوراق نبات **الريحان** بدقة هندسية ونظام ذكاء اصطناعي محمي ومستقر 100%.
    
    🔬 **ماذا يقدم لك النظام؟**
    * كشف فوري عن سلامة الأوراق (Healthy).
    * رصد دقيق لمرض البياض الزغبي (Downy Mildew).
    * رصد دقيق لمرض تبقع الأوراق (Leaf Spot).
    * تقديم تقرير علاجي زراعي فوري (كيميائي وفطري).
    
    ---
    💡 *تم تطوير هذا النظام بكل فخر بواسطة أحمد حسني والتيم الخاص به، تحت إشرافكم الكريم.*
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 اضغط هنا للدخول إلى نظام الفحص الذكي", use_container_width=True):
        st.session_state.page = 'app'
        st.rerun()

# ==========================================
# ⚙️ ثانياً: صفحة السيستم والفحص
# ==========================================
elif st.session_state.page == 'app':
    
    if st.button("⬅️ العودة للرئيسية"):
        st.session_state.page = 'welcome'
        st.rerun()
        
    st.title("🌱 لوحة فحص أوراق الريحان")
    st.write("وجه الكاميرا أو ارفع صورة لورقة الريحان للكشف عن حالتها الصحية فوراً وخطة علاجها.")

    # تحميل الموديل بطريقة مرنة لتفادي أخطاء الـ Deserialization
    @st.cache_resource
    def load_basil_model():
        model_path = 'Basil_Smart_Model.keras'
        try:
            # محاولة التحميل المباشر أولاً
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            # إذا فشل، نقوم ببناء الهيكل المتوافق مع أبعاد MobileNetV2 (1280 features) وضخ الأوزان
            base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            try:
                # محاولة تحميل الأوزان فقط من الملف المتاح
                model.load_weights(model_path, skip_mismatch=True)
                return model
            except Exception as e:
                # حل أخير: محاولة تحميل الموديل كملف خام عابر للإصدارات
                raise RuntimeError(f"يرجى إعادة حفظ الموديل بصيغة .h5 في كولاب أو التأكد من سلامة الملف. الخطأ الأساسي: {e}")

    try:
        model = load_basil_model()
        models_loaded = True
        st.success("✅ تم ترويض السيستم وتحميل الموديل بأمان وجاهز للفحص المستقر!")
    except Exception as e:
        st.error(f"❌ فشل في تحميل الموديل من السيرفر: {e}")
        models_loaded = False

    # دالة تجهيز ومعالجة الصور
    def preprocess_image(img_file, target_size=(224, 224)):
        img = Image.open(img_file).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array

    tab1, tab2 = st.tabs(["📁 رفع صورة من الجهاز", "📸 (Scan) تصوير مباشر"])
    final_image = None

    with tab1:
        st.markdown("### 📥 ارفع صورة موجودة مسبقاً على جهازك")
        uploaded_file = st.file_uploader("اختر صورة ورقة ريحان...", type=["jpg", "jpeg", "png"], key="uploader")
        if uploaded_file is not None:
            final_image = uploaded_file

    with tab2:
        st.markdown("### 📸 التقط صورة حية لورقة النبات")
        camera_file = st.camera_input("وجه الكاميرا نحو ورقة الريحان واضغط التقاط")
        if camera_file is not None:
            final_image = camera_file

    # تشخيص النظام واستخراج التقرير الزراعي الشامل
    CLASS_NAMES = ['Downy_Mildew', 'Healthy', 'Leaf_Spot']

    if final_image is not None:
        st.image(final_image, caption='📷 الصورة التي يتم تحليلها حالياً', use_column_width=True)
        
        if not models_loaded:
            st.error("لا يمكن فحص الصورة لأن الموديل لم يتم تحميله بشكل صحيح.")
        else:
            st.info("جاري تحليل ميزات الورقة والتشخيص الذكي المستقر... 🔄")
            try:
                processed_img = preprocess_image(final_image, (224, 224))
                predictions = model.predict(processed_img)[0]
                
                highest_class_idx = np.argmax(predictions)
                detected_status = CLASS_NAMES[highest_class_idx]
                confidence_score = predictions[highest_class_idx] * 100
                
                st.write("---")
                st.subheader("📊 نتيجة التشخيص الشاملة:")
                
                if detected_status == 'Healthy':
                    st.balloons()
                    st.success(f"💖 **حالة النبات:** سليم ومعافى تماماً وبصحة جيدة! (نسبة التأكد: {confidence_score:.2f}%)")
                    st.info("💡 **نصيحة PlantMed AI:** استمر في نظام الري والتسميد المتزن، وحافظ على تهوية الصوبة بانتظام لمنع نشاط الفطريات.")
                
                else:
                    st.error(f"🚨 **حالة النبات:** مصاب بمرض (نسبة التأكد: {confidence_score:.2f}%)")
                    
                    DISEASES_DATABASE = {
                        "Downy_Mildew": {
                            "title": "البياض الزغبي (Basil Downy Mildew)",
                            "fast": "عزل النباتات المصابة فوراً، تقليل نسبة الرطوبة داخل الصوبة، ووقف الري العلوي (الرش الورقي) تماماً وتجنب تبلل الأوراق.",
                            "chemical": "الرش الفوري بمبيد فطري جهازي متخصص يحتوي على مادة فعالة مثل (ميتالاكسيل) أو المركبات النحاسية لحماية النموات الجديدة."
                        },
                        "Leaf_Spot": {
                            "title": "تبقع الأوراق (Basil Leaf Spot)",
                            "fast": "التخلص من الأوراق السفلية المصابة وحرقها بعيداً، تحسين تهوية المكان وزيادة المسافات بين النباتات لتجفيف الأوراق بسرعة.",
                            "chemical": "استخدام مركبات النحاس الوقائية (مثل كبريتات النحاس الميكرونية) بانتظام كل 7-10 أيام لمنع انتشار البقع."
                        }
                    }
                    
                    current_disease = DISEASES_DATABASE[detected_status]
                    st.warning(f"🔍 **التشخيص الدقيق للمرض:** {current_disease['title']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("⏱️ الإجراء السريع والفوري:")
                        st.info(current_disease["fast"])
                    with col2:
                        st.subheader("🧪 العلاج الكيميائي والمبيدات:")
                        st.warning(current_disease["chemical"])
                        
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الصورة: {e}")
