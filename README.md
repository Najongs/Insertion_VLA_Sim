# Insertion_VLA_Sim
Insertion_VLA_Sim

## Sim 만드는 방법
일단 로봇은 MECA를 활용한다. 따라서 URDF/Meca500_urdf.urdf에서 로봇의 형상이 생성되고 이를 URDF/Insertion_task.xacro에서 불러온다.
이후에 URDF/EYE_PHAMTOM.SLDASM.urdf에서 눈 모형에 대한 정보와 추가 배경의 정보들을 URDF/Insertion_task.xacro에 종합하여 작성하였다.

이후 이를 통합본 urdf파일로 변환하기 위해서 xacro Insertion_task.xacro > combined_model.urdf 로 변환하고 만들어진 통합본으로 .xml 파일을 생성한다.

C:\Users\DGIST\Desktop\mujoco-3.4.0-windows-x86_64\bin\compile.exe combined_model.urdf meca_scene22.xml

해당 방법을 사용하면 기본적인 배경과 내용만이 담긴 파일이 생성되는데 일부 액츄에이터나, 환경정보, 카메라 등 일부가 담기지 않으므로 URDF/implementation_code.sh에 추가되는 정보가 담긴 백업을 마련해두었다. 

색상, 텍스쳐 표현을 위해서 textures와 STL 폴더에 내용이 저장 되어있으며, 시뮬레이션 생성과정 랜덤 배경을 위한 random_backgrounds가 존재한다.

예제 동작을 실행하기 위해 final.py로 실제 동작(바늘 삽입)을 확인할 수 있으며, 데이터 생성은 Save_dataset.py로 clean버전, Save_dataset_arg.py로 domain adaptation버전을 생성한다.

추가로 병렬생성을 위한 코드 run_parallel_nodomain.py, run_parallel_simple.py이 존재한다.

데이터를 확인하기 위한 data_replay.py도 존재한다.