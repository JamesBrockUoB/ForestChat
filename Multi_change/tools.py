# 然后直接导入文件中的函数
from predict import Change_Perception

if __name__ == "__main__":
    Change_Perception = Change_Perception()
    imgA_path = "./data/LEVIR-MCI-Trees-dataset/images/test/A/test_000004_A.png"
    imgB_path = "./data/LEVIR-MCI-Trees-dataset/images/test/B/test_000004_B.png"
    savepath_mask = "./predict_results/4_mask.png"
    # Change_Perception.generate_change_caption(imgA_path, imgB_path)
    mask = Change_Perception.change_detection(imgA_path, imgB_path, savepath_mask)
    num = Change_Perception.compute_object_num(mask, "road")
    percentage = Change_Perception.compute_change_percentage(savepath_mask)
    print(num)
    print(percentage)
