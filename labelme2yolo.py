from general_json2yolo import convert_coco_json
import labelme2coco

print("="*20, ' Start transform... ', "="*20)
# 'input_dir' 에 데이터 원본(이미지), labelme로 만들어진 (이미지파일명).json 파일이 있는 경로
# 'output_dir' 에 1차 결과 파일들 (== conver_coco_json 의 인풋경로와 같아야함.)
# 'labels' 에 사전기재된 레이블링 이름들.
#  .txt파일 0번과 1번에 위치할 인덱스는
#  __ignore__
# _background_ 로 시작하도록

labelme2coco.run(input_dir='/path/to/datasets/images/', output_dir='result', labels='labels.txt', noviz=True)

convert_coco_json('./result/',  # directory with *.json
                  use_segments=True,
                  cls91to80=True)
print("="*20, ' Done!. ', "="*20)
