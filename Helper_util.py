import string
import easyocr
import cv2
# OCR reader:
# The following easyocr instance reads arabic and english language.
# reader = easyocr.Reader(['en'])

# Mapping dictionaries for character conversion
# As we know the format of the targeted license plate, we'll essentially help the model with detecting accurately.
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the result to a CSV file.

    Args:
        results (dict): Dictionary containing the result.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_c', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False # Edit this later

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


# Extra func to test for Saudi license plates.
# # Define Arabic letters if needed (you can expand this list as needed)
# arabic_letters = 'أبجدھوزيكلمنسعفصقرفشتثخضصظطظعغ'

# # Define character-to-integer and integer-to-character mappings if applicable
# dict_int_to_char = {str(i): chr(65 + i) for i in range(10)}  # Example mapping
# dict_char_to_int = {chr(65 + i): str(i) for i in range(10)}  # Example mapping

# def license_complies_format_SA(text):
#     """
#     Check if the license plate text complies with the Saudi Arabian format.

#     Args:
#         text (str): License plate text.

#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     # Saudi license plates generally have the format: XXX-9999
#     if len(text) != 8:
#         return False

#     # Check format: XXX-9999
#     if not (text[3] == '-' and
#             all(c in (string.ascii_uppercase + arabic_letters) for c in text[:3]) and
#             all(c in string.digits for c in text[4:])):
#         return False

#     return True

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


# def format_license_SA(text):
#     """
#     Format the Saudi license plate text by converting characters using the mapping dictionaries.

#     Args:
#         text (str): License plate text.

#     Returns:
#         str: Formatted license plate text.
#     """
#     license_plate_ = ''

#     # Define mappings
#     mapping = {
#         0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,  # Letters to digits
#         4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,  # Letters to digits
#         2: dict_char_to_int, 3: dict_char_to_int,  # Letters to digits (if needed)
#     }

#     for j in range(len(text)):
#         if j in mapping.keys():
#             if text[j] in mapping[j]:
#                 license_plate_ += mapping[j][text[j]]
#             else:
#                 license_plate_ += text[j]
#         else:
#             license_plate_ += text[j]

#     return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped threshed image.

    Args:
        license_plate_crop (Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


# def assign_car(license_plate, vehicle_track_ids):
#     """
#     Retrieve the vehicle coordinates and ID based on the license plate coordinates.
#
#     Args:
#         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
#         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
#
#     Returns:
#         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
#     """
#     x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate
#
#     foundIt = False
#     for j in range(len(vehicle_track_ids)):
#         x1_v, y1_v, x2_v, y2_v, track_id, class_id_v = vehicle_track_ids[j]
#
#         if x1_lp > x1_v and y1_lp > y1_v and x2_lp < x2_v and y2_lp < y2_v:
#             car_indx = j
#             foundIt = True  # If the car's license plate was found then break and assign it.
#             break
#
#     if foundIt:
#         # Return the vehicle's bounding box and ID
#         return vehicle_track_ids[car_indx][:5]
#     # Return None if no matching vehicle is found
#     return -1, -1, -1, -1, -1

# More loose assign_car:
def assign_car(license_plate, vehicle_track_ids, tolerance=0.1):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates with a margin of tolerance.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
        tolerance (float): Percentage of tolerance to make the bounding box check more lenient. Default is 10%.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        x1_v, y1_v, x2_v, y2_v, track_id, class_id_v = vehicle_track_ids[j]

        # Calculate the tolerance as a fraction of the vehicle's width and height
        width_v = x2_v - x1_v
        height_v = y2_v - y1_v
        tol_w = tolerance * width_v
        tol_h = tolerance * height_v

        # Check if the license plate is within the vehicle's bounding box with some tolerance
        if (x1_lp > (x1_v - tol_w) and y1_lp > (y1_v - tol_h) and
            x2_lp < (x2_v + tol_w) and y2_lp < (y2_v + tol_h)):
            car_indx = j
            foundIt = True  # If the car's license plate was found then break and assign it.
            break

    if foundIt:
        # Return the vehicle's bounding box and ID
        return vehicle_track_ids[car_indx][:5]
    # Return None if no matching vehicle is found
    return -1, -1, -1, -1, -1


def annotate_image(image, results):
    for car_id, data in results.items():
        if 'car' in data:
            car_bbox = data['car']['bbox']
            cv2.rectangle(image, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])),
                          (0, 255, 0), 2)

        if 'license_plate' in data:
            plate_bbox = data['license_plate']['bbox']
            cv2.rectangle(image, (int(plate_bbox[0]), int(plate_bbox[1])), (int(plate_bbox[2]), int(plate_bbox[3])),
                          (0, 0, 255), 2)
            cv2.putText(image, data['license_plate']['text'], (int(plate_bbox[0]), int(plate_bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return image