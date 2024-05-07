"""
A script for detecting and labeling restricted content in JSON-formatted text files.

This script employs a pre-trained language model from the transformers library to analyze textual content for the presence of restricted topics. It updates each JSON file with a label indicating whether restricted content was detected and the confidence level of the prediction.

Functions:
    get_response(text, tokenizer, model): Generates a response from the language model given an input text.
    parse_model_response(model_response): Parses the model's response to extract a numerical label indicating the presence of restricted content.
    check_restricted_content(text): Analyzes the provided text for restricted content based on predefined criteria and returns a binary label.
    predict_with_confidence(text, iterations): Repeats the restricted content check multiple times to determine the most common label and its confidence level.
    update_json_file(args): Processes a single JSON file, analyzes its text content for restricted topics, and updates the file with a label and confidence score.
    update_json_files(source_dir, target_dir): Iterates over JSON files in the specified source directory, applies content analysis, and saves the updated files to the target directory.

Main Process:
    1. The script checks for the availability of a CUDA-compatible GPU and selects the appropriate device (GPU or CPU) for model inference.
    2. It initializes the tokenizer and model from a specified model path, ensuring the model is loaded onto the appropriate device.
    3. The main execution block (`if __name__ == "__main__":`) specifies the source and target directories for the JSON files to be processed.
    4. Each JSON file in the source directory is read, and its text content is analyzed for restricted topics.
    5. The script updates each file with a 'label' key indicating the presence of restricted content (0 or 1) and a 'confidence' key representing the confidence level of the prediction.
    6. Updated JSON files are saved to the target directory, preserving the original file names.
    7. The script reports the total time taken to process all files upon completion.

Usage:
    This script is intended to be executed directly. User must define the source and target directories for JSON files before running the script. It is suitable for batch processing of text content for content moderation purposes.

Note:
    The script assumes that each JSON file contains a 'text_chunk' key with textual content to be analyzed. It requires an internet connection for downloading model weights upon the first run.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import os
import json
import time
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "inception-mbzuai/jais-13b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)


def get_response(text, tokenizer=tokenizer, model=model):
    """
    Generates a response given an input text using the specified tokenizer and model.

    Args:
        text (str): The input text for which a response is generated.
        tokenizer: The tokenizer used to tokenize the input text.
        model: The model used to generate the response.

    Returns:
        str: The generated response.

    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=1,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response: [|AI|]")
    return response


def parse_model_response(model_response):
    """
    Parses the model response and returns a label.

    Args:
        model_response (str): The response from the model.

    Returns:
        int: The label extracted from the model response. Returns 0 if "0" is found in the response,
        returns 1 if "1" is found in the response, otherwise returns 1.
    """
    cleaned_response = model_response.strip()
    if "0" in cleaned_response:
        return 0
    elif "1" in cleaned_response:
        return 1
    else:
        return 1


def check_restricted_content(text):
    """
    Check if the given text contains any restricted topics.

    Args:
        text (str): The text to be analyzed.

    Returns:
        int: A binary value indicating whether the text contains any restricted topics.
            - If the text does not contain any restricted topics, return 0.
            - If the text contains any restricted topics, return 1.
    """
    prompt = """
    ### Instruction:
    قم بتحليل النص المقدم بعمق لتحديد ما إذا كان يحتوي على مواضيع محظورة. فيما يلي أمثلة مفصلة وشروحات للمواضيع التي تعتبر محظورة. فكر بعمق في سياق ودلالات كل جانب مذكور. يجب أن تكون استجابتك رقمًا واحدًا فقط، "0" أو "1"، دون تقديم أي نص أو رأي إضافي.
        
    1. **المحتوى الديني الحساس**: يشمل الإساءة إلى الديانات، السخرية من النصوص الدينية، الرموز، الطقوس، أو الشخصيات الدينية، وترويج الكراهية الدينية أو الطائفية.
    2. **اللعن أو الكلام البذيء**: استخدام اللغة الفاحشة، السباب، أو الكلام الذي يُعتبر مهينًا أو مسيئًا بشكل واضح.
    3. **المحتوى الجنسي**: الوصف الصريح للأفعال الجنسية، المواد الإباحية، أو المحتوى الذي يروج للاستغلال الجنسي أو غير الملائم.
    4. **الحساسية السياسية**: المحتوى الذي قد يثير الاضطرابات السياسية، ينتقد الحكومات أو الشخصيات السياسية بطريقة مثيرة للجدل، أو يعتبر دعاية سياسية.
    5. **العنف وخطاب الكراهية**: ترويج العنف، الإرهاب، أو الكراهية ضد الأفراد أو الجماعات على أساس العرق، الدين، الجنس، الهوية الجنسية، أو أي خصائص أخرى.
    6. **الأنشطة غير القانونية**: تشجيع أو ترويج السلوكيات أو الأنشطة غير القانونية مثل المخدرات، السرقة، الاحتيال، أو القرصنة.
    7. **المحتوى العنصري أو الجنسي**: المحتوى الذي يروج للتمييز، يعزز الصور النمطية الضارة، أو يستخدم الإهانات العنصرية أو الجنسية.
    8. **المحتوى الضار أو الخطير**: ترويج الأذى الذاتي، التحديات الخطيرة، الإرشادات لإنشاء الأسلحة، أو أي ممارسات غير آمنة.  
    
    في نهاية التحليل، قدم إجابة تتكون من رقم واحد فقط، بناءً على معايير المواضيع المحظورة المذكورة:
    - إذا لم يحتوي النص على أي من المواضيع المحظورة، قدم "0".
    - إذا احتوى النص على أي من المواضيع المحظورة، قدم "1".

    ### Note: If you are not sure, return 1 but first make a deep analysis for the text.
        
    ### Input: [|Human|] {text}
    ### Response: [|AI|]
    """.format(text=text)

    model_response = get_response(prompt)
    return parse_model_response(model_response[1])


def predict_with_confidence(text, iterations=5):
    """
    Repeats the restricted content check multiple times to determine the most common label and its confidence level.
    
    Args:
        text (str): The text to be analyzed for restricted content.
        iterations (int, optional): The number of times to repeat the content analysis. Defaults to 5.

    Returns:
        tuple: A tuple containing the most common label (0 or 1) and its confidence level.

    """
    predictions = []
    for _ in tqdm(range(iterations), desc="Predicting"):
        predictions.append(check_restricted_content(text))
    prediction_count = Counter(predictions)

    most_common_label, most_common_count = prediction_count.most_common(1)[0]
    confidence = most_common_count / iterations
    return most_common_label, confidence


def update_json_file(args):
    """
    Update a JSON file with a label and confidence score based on the text content.

    Args:
        args (tuple): A tuple containing the source directory, target directory, and filename.

    Returns:
        None
    """
    source_dir, target_dir, filename = args
    file_path = os.path.join(source_dir, filename)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Skipping empty or non-existent file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            text = data['text_chunk']
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return

        label, confidence = predict_with_confidence(text)
        data['label'] = label
        data['confidence'] = confidence

        target_file_path = os.path.join(target_dir, filename)
        with open(target_file_path, 'w', encoding='utf-8') as target_file:
            target_file.write(json.dumps(data, ensure_ascii=False, indent=4))


def update_json_files(source_dir, target_dir):
    """
    Update JSON files from the source directory to the target directory.

    Args:
        source_dir (str): The path to the source directory containing the JSON files.
        target_dir (str): The path to the target directory where the updated JSON files will be saved.

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    filenames = [filename for filename in os.listdir(source_dir) if filename.endswith('.json')]
    for filename in tqdm(filenames, desc="Updating JSON files"):
        update_json_file((source_dir, target_dir, filename))


if __name__ == "__main__":
    source_directory = './src/nlp/restricted_topic_detection/renamed_dataset'
    target_directory = './src/nlp/restricted_topic_detection/labeled_dataset'
    start_time = time.time()
    update_json_files(source_directory, target_directory)
    print("Finihsed!")
    print(f"Labeling completed in {time.time() - start_time:.2f} seconds.")
