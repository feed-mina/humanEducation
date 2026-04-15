import os
from PIL import Image
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from config import Config


# In[2]:


class ImageHandler:
    @staticmethod
    def process_image(image):
        if image is None:
            return None
        
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            return None

    def generate_wordcloud(self):
        try:
            with open(Config.MAIN_DOC_PATH, 'r', encoding='utf-8') as f:
                text = f.read()

            wordcloud = WordCloud(
                font_path=Config.FONT_PATH,
                background_color='white',
                width=800, height=400
            ).generate(text)

            os.makedirs(os.path.dirname(Config.IMG_PATH), exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(Config.IMG_PATH)
            plt.close()

            return Config.IMG_PATH
        except Exception as e:
            return f"오류 발생: {e}"