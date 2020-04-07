import os
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras import backend as K
from os import listdir
K.clear_session()

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
im = ''
result = '...'
percentage = '...'
i = 0
imageName = ''
solution = ''
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    global im, result, percentage , i , imageName , solution
    target = os.path.join(APP_ROOT, 'static\\')
    print(f'Target : {target}')

    if not os.path.isdir(target):
        os.mkdir(target)
    for imgg in os.listdir(target):
        try:
            imgPath = target + imgg
            os.remove(imgPath)
            print(f'Removed : {imgPath}')
        except Exception as e:
            print(e)
        
    for file in request.files.getlist("file"):
        print(f'File : {file}')
        i += 1
        imageName = str(i) + '.JPG'
        filename = file.filename
        destination = "/".join([target, imageName])
        print(f'Destination : {destination}')
        file.save(destination)
        print('analysing Image')
        try:
            image = os.listdir('static')
            im = destination
            print(f'Analysing Image : {im}')
        except Exception as e:
            print(e)
        result = "Failed to Analyse"
        percentage = "0 %"
        try:
            detect()
            solution = solutions(result)
        except Exception as e:
            print(f'Error While Loading : {e}')  
    return render_template('complete.html', name=result, accuracy=percentage , img = imageName , soln = solution)


def detect():
    global im, result, percentage
    print(f'Image : {im}')
    # resolution
    ht=50
    wd=50
    classNames = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy" , "Potato___Early_blight" , "Potato___healthy" ,  "Potato___Late_blight" ,
        "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy",
                  "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
                  "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
                  "Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus"]
    totClass = len(classNames)
    print(classNames)
    print(totClass)
    mdl = r"LeafDisease50x50.h5"
    image = cv2.imread(im)
    orig = image.copy()
    try:
        image = cv2.resize(image, (ht, wd))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        print("Error Occured : ",e)
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(mdl)
    (zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen) = model.predict(image)[0]
    prob = [zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen]

    maxProb = max(prob)
    maxIndex = prob.index(maxProb)
    label = classNames[maxIndex]
    proba = maxProb
    result = label
    percentage = float("{0:.2f}".format(proba * 100))
    for i in range(0,totClass):
        print(f'{classNames[i]} : {prob[i]}')

Tomato_Bacterial_spot = """
Fertilizers:
1. Bonide Citrus, Fruit & Nut Orchard Spray (32 Oz)
2. Bonide Infuse Systemic Fungicide...
3. Hi-Yield Captan 50W fungicide (1...
4. Monterey Neem Oil

"""

Tomato_Early_blight = """
\n
1. Mancozeb Flowable with Zinc Fungicide Concentrate
2. Spectracide Immunox Multi-Purpose Fungicide Spray Concentrate For Gardens
3. Southern Ag – Liquid Copper Fungicide
4. Bonide 811 Copper 4E Fungicide
5. Daconil Fungicide Concentrate. 

"""
Tomato_healthy = """
\nYour Plant Is Healthier.
"""
Tomato_Late_blight = """
\n
Plant resistant cultivars when available.
Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.
Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation.
Destroy all tomato and potato debris after harvest.
"""
Tomato_Leaf_Mold = """
\nFungicides : 
1. Difenoconazole and Cyprodinil
2. Difenoconazole and Mandipropamid
3. Cymoxanil and Famoxadone
4. Azoxystrobin and Difenoconazole

"""
Tomato_Septoria_leaf_spot = """
\n
Use disease-free seed and dont save seeds of infected plants
Start with a clean garden by disposing all affected plants.
Water aids the spread of Septoria leaf spot. Keep it off the leaves as much as possible by watering at the base of the plant only. 
Provide room for air circulation. Leave some space between your tomato plants so there is good airflow.

"""
Tomato_Spider_mites_Two_spotted_spider_mite = """
\n
Prune leaves, stems and other infested parts of plants well past any webbing and discard in trash (and not in compost piles). Don’t be hesitant to pull entire plants to prevent the mites spreading to its neighbors.
Use the Bug Blaster to wash plants with a strong stream of water and reduce pest numbers.
Commercially available beneficial insects, such as ladybugs, lacewing and predatory mites are important natural enemies. For best results, make releases when pest levels are low to medium.
Dust on leaves, branches and fruit encourages mites. A mid-season hosing (or two!) to remove dust from trees is a worthwhile preventative.
Insecticidal soap or botanical insecticides can be used to spot treat heavily infested areas.
"""
Tomato__Target_Spot = """
1. Remove old plant debris at the end of the growing season; otherwise, the spores will travel from debris to newly planted tomatoes in the following growing fc, thus beginning the disease anew. Dispose of the debris properly and don’t place it on your compost pile unless you’re sure your compost gets hot enough to kill the spores.

2. Rotate crops and don’t plant tomatoes in areas where other disease-prone plants have been located in the past year – primarily eggplant, peppers, potatoes or, of course – tomatoes. Rutgers University Extension recommends a three-year rotation cycle to reduce soil-borne fungi.

3. Pay careful attention to air circulation, as target spot of tomato thrives in humid conditions. Grow the plants in full sunlight. Be sure the plants aren’t crowded and that each tomato has plenty of air circulation. Cage or stake tomato plants to keep the plants above the soil.

4. Water tomato plants in the morning so the leaves have time to dry. Water at the base of the plant or use a soaker hose or drip system to keep the leaves dry. Apply a mulch to keep the fruit from coming in direct contact with the soil. Limit to mulch to 3 inches or less if your plants are bothered by slugs or snails.

5. You can also apply fungal spray as a preventive measure early in the season, or as soon as the disease is noticed.

"""
Tomato__Tomato_mosaic_virus = """
\n
Fungicides will not treat this viral disease.
Avoid working in the garden during damp conditions (viruses are easily spread when plants are wet).
Frequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc. 
Remove and destroy all infected plants.Do not compost.
Do not save seed from infected crops.
"""
Tomato__Tomato_YellowLeaf__Curl_Virus = """
\n
Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers. 
Cover plants with floating row covers of fine mesh (Agryl or Agribon) to protect from whitefly infestations.
Practice good weed management in and around fields to the extent feasible.
Remove and destroy old crop residue and volunteers on a regional basis.
"""
def solutions(disease):
    switcher = {
        "Tomato_Bacterial_spot": Tomato_Bacterial_spot ,
        "Tomato_Early_blight": Tomato_Early_blight ,
        "Tomato_healthy": Tomato_healthy , 
        "Tomato_Late_blight" : Tomato_Late_blight,
        "Tomato_Leaf_Mold" : Tomato_Leaf_Mold,
        "Tomato_Septoria_leaf_spot" : Tomato_Septoria_leaf_spot,
        "Tomato_Spider_mites_Two_spotted_spider_mite" : Tomato_Spider_mites_Two_spotted_spider_mite,
        "Tomato__Target_Spot" : Tomato__Target_Spot,
        "Tomato__Tomato_mosaic_virus" : Tomato__Tomato_mosaic_virus,
        "Tomato__Tomato_YellowLeaf__Curl_Virus" : Tomato__Tomato_YellowLeaf__Curl_Virus,
        }
    return switcher.get(disease,"Not Found In The List")
        
if __name__ == "__main__":
    app.run(port=4555, debug=True)
