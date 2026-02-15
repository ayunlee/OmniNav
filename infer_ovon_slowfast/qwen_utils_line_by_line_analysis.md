# qwen_utils.py 한 줄씩 상세 분석

- 대상 파일: `infer_ovon_slowfast/qwen_utils.py`
- 설명 방식: **원본 파일의 모든 줄(빈 줄 포함)**을 라인 번호 기준으로 해설

| 라인 | 원문 | 상세 설명 |
|---:|---|---|
| 1 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 2 | `import numpy as np` | 필요한 모듈 `numpy as np` 를 현재 네임스페이스에 로드합니다. |
| 3 | `from PIL import Image        ` | `from PIL import Image` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 4 | `from qwen_vl_utils import smart_resize` | `from qwen_vl_utils import smart_resize` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 5 | `min_pixels = 56 * 56` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 6 | `max_pixels = 4480 * 4480` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 7 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 8 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 9 | `import numpy as np` | 필요한 모듈 `numpy as np` 를 현재 네임스페이스에 로드합니다. |
| 10 | `from scipy.spatial.transform import Rotation as R` | `from scipy.spatial.transform import Rotation as R` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 11 | `import re` | 필요한 모듈 `re` 를 현재 네임스페이스에 로드합니다. |
| 12 | `import json` | 필요한 모듈 `json` 를 현재 네임스페이스에 로드합니다. |
| 13 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 14 | `def parse_response(response_text):` | 함수 `parse_response` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 15 | `    pattern = re.compile(r"coordinate (\[.*?\])")` | 정규식 패턴 객체를 준비해 응답 텍스트에서 좌표 문자열을 추출할 준비를 합니다. |
| 16 | `    match = pattern.search(response_text)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 17 | `    if match:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 18 | `        coord_str = match.group(1)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 19 | `        try:` | 예외 가능 구간을 감싸는 `try` 블록 시작입니다. |
| 20 | `            coord_list = json.loads(coord_str)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 21 | `            ` | 가독성을 위해 구분한 빈 줄입니다. |
| 22 | `            if isinstance(coord_list, list):` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 23 | `                if "found" in response_text:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 24 | `                    return coord_list, True` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 25 | `                else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 26 | `                    return coord_list, False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 27 | `            else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 28 | `                return None, False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 29 | `        except json.JSONDecodeError:` | 지정한 예외가 발생했을 때의 복구/우회 로직 분기입니다. |
| 30 | `            return None, False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 31 | `    return None, False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 32 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 33 | `def process_vision(colorlist):` | 함수 `process_vision` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 34 | `    """处理消息中的图像和视频信息"""` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 35 | `    image_inputs = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 36 | `    video_inputs = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 37 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 38 | `    for image in colorlist: ` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 39 | `            image = Image.fromarray(image)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 40 | `            image = image.resize((486,420))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 41 | `            image_inputs.append(image)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 42 | `                        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 43 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 44 | `    return image_inputs, video_inputs` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 45 | `import numpy as np` | 필요한 모듈 `numpy as np` 를 현재 네임스페이스에 로드합니다. |
| 46 | `import quaternion` | 필요한 모듈 `quaternion` 를 현재 네임스페이스에 로드합니다. |
| 47 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 48 | `from scipy.spatial.transform import Rotation as R` | `from scipy.spatial.transform import Rotation as R` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 49 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 50 | `def transform_to_local_frame(world_point, agent_world_coord, agent_world_quat):` | 함수 `transform_to_local_frame` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 51 | `    world_point = np.array(world_point)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 52 | `    agent_world_coord = np.array(agent_world_coord)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 53 | `    relative_point = world_point - agent_world_coord` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 54 | `    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 55 | `    agent_rotation = R.from_quat(quat_xyzw)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 56 | `    local_point = agent_rotation.apply(relative_point, inverse=True)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 57 | `    local_point = local_point.round(2).tolist()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 58 | `    res = [local_point[0] ,local_point[1]* -1, local_point[2] * -1]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 59 | `    return res` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 60 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 61 | `def transform_from_local_frame(local_point, agent_world_coord, agent_world_quat):` | 함수 `transform_from_local_frame` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 62 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 63 | `    local_point = np.array([` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 64 | `        local_point[0] ,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 65 | `        local_point[1]* -1,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 66 | `        local_point[2] * -1` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 67 | `    ])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 68 | `    agent_world_coord = np.array(agent_world_coord)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 69 | `    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 70 | `    agent_rotation = R.from_quat(quat_xyzw)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 71 | `    relative_point_in_world = agent_rotation.apply(local_point)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 72 | `    reconstructed_world_point = relative_point_in_world + agent_world_coord` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 73 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 74 | `    return reconstructed_world_point` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 75 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 76 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 77 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 78 | `obj_goal_template = [` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 79 | `"Find a {} in your immediate surroundings and stop when you see one.", ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 80 | `"Explore the area until you locate a {}. Stop when you've reached its location.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 81 | `"Move through the environment to discover a {}. Your task is complete when you're directly facing it.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 82 | `"Navigate to any visible {}. Stop immediately upon successful discovery.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 83 | `"Search for an instance of {} within this space. Terminate navigation once you've positioned yourself within arm's reach of it.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 84 | `"Survey the surroundings until you identify a {}. Stop navigating as soon as you are positioned directly in front of it",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 85 | `"Roam through the space until a {} is spotted. Terminate navigation the moment you’re certain you’re facing it.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 86 | `"Go to the {}, then stop at the front of it.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 87 | `"Move to the nearst {}, then stop",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 88 | `"Navigate to a nearst {}, then stop over there.",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 89 | `"Get close to the {}, then stop",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 90 | `"Could you help me find a {}? Show me the way"]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 91 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 92 | `import random` | 필요한 모듈 `random` 를 현재 네임스페이스에 로드합니다. |
| 93 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 94 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 95 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 96 | `def getresult(qwen, processor,bank, current_frontiers, decision_agent_state, object_category, decision_num,` | 함수 `getresult` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 97 | `               visited_frontier_set):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 98 | `    ref_coord = decision_agent_state.position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 99 | `    ref_quat_dict = {` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 100 | `        'x': decision_agent_state.rotation.x, 'y': decision_agent_state.rotation.y,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 101 | `        'z': decision_agent_state.rotation.z, 'w': decision_agent_state.rotation.w` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 102 | `    }` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 103 | `    all_prompt_images = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 104 | `    spin_image,spin_state=bank.get_spin_data()  ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 105 | `    all_prompt_images.extend(spin_image)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 106 | `    spin_images_content ="1: These four images show a 360-degree panoramic view around Observer's perspective,position is all [0.00,0.00], taken at 90-degree intervals: <image><image><image><image>"` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 107 | `    main_images_content ="2: This is the reference image from Observer's perspective for all coordinates: <image>"` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 108 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 109 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 110 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 111 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 112 | `    local_to_global_map = {}` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 113 | `    frontier_parts = ['3: The coordinates of the explorable frontiers are: ']` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 114 | `    for frontier_coord in current_frontiers:` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 115 | `        local_coord = transform_to_local_frame(frontier_coord, ref_coord, ref_quat_dict)` | 월드 좌표를 에이전트 기준 로컬 좌표계로 변환하는 유틸리티를 호출하거나 정의합니다. |
| 116 | `        x, z = local_coord[0], local_coord[2]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 117 | `        local_to_global_map[(x, z)] = frontier_coord` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 118 | `        frontier_parts.append(f"[{x:.2f}, {z:.2f}]")` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 119 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 120 | `    frontier_content = ''.join(frontier_parts)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 121 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 122 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 123 | `    task_sentence = "instruction: "+random.choice(obj_goal_template).format(object_category)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 124 | `    user_content = "\n".join([spin_images_content,main_images_content,frontier_content,task_sentence])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 125 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 126 | `    current_messages=[]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 127 | ` ` | 가독성을 위해 구분한 빈 줄입니다. |
| 128 | `    msg_copy={}` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 129 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 130 | `    if len(current_messages) == 0:  ` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 131 | `        content = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 132 | `        content.append({"type": "image", "image": all_prompt_images})` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 133 | `        content.append({"type": "text", "text":user_content})` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 134 | `        msg_copy['content'] = content` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 135 | `        current_messages.append(msg_copy)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 136 | `    current_messages[0]['role']='user'` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 137 | `    text = processor.apply_chat_template(` | Qwen 프로세서의 채팅 템플릿으로 멀티모달 입력을 모델이 이해하는 프롬프트 문자열로 변환합니다. |
| 138 | `        current_messages, tokenize=False, add_generation_prompt=True` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 139 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 140 | `    text=text.replace('<\|vision_start\|><\|image_pad\|><\|vision_end\|>','')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 141 | `    text=text.replace('<image>','<\|vision_start\|><\|image_pad\|><\|vision_end\|>')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 142 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 143 | `    image_inputs, video_inputs = process_vision(all_prompt_images)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 144 | `    inputs = processor(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 145 | `        text=text,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 146 | `        images=image_inputs,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 147 | `        padding=True,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 148 | `        padding_side="left",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 149 | `        return_tensors="pt",` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 150 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 151 | `    device = qwen.device ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 152 | `    inputs = inputs.to(device)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 153 | `    print(text.count('image_pad'))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 154 | `    with torch.no_grad():` | 컨텍스트 매니저를 사용해 리소스를 안전하게 열고 자동 정리합니다. |
| 155 | `        generated_ids = qwen.generate(**inputs, max_new_tokens=512)` | 언어-비전 모델의 생성 API를 호출해 다음 토큰(응답)을 생성합니다. |
| 156 | `        generated_ids_trimmed = [` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 157 | `            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 158 | `        ]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 159 | `        output_texts = processor.batch_decode(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 160 | `            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 161 | `        )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 162 | `        parsed_local_target,is_final_decision=parse_response(output_texts[0])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 163 | `    print(output_texts)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 164 | `    if parsed_local_target is None:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 165 | `        if  local_to_global_map is None and current_frontiers is not None:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 166 | `            return current_frontiers[0] if current_frontiers else None, False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 167 | `    model_choice_coord = np.array(parsed_local_target)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 168 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 169 | `    min_dist = float('inf')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 170 | `    best_match_global_coord = None` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 171 | `    if is_final_decision:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 172 | `        model_choice_coord= np.insert(model_choice_coord, 1, 0)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 173 | `        best_match_global_coord=transform_from_local_frame(model_choice_coord,ref_coord, ref_quat_dict)` | 로컬 좌표를 다시 월드 좌표로 복원하는 유틸리티를 호출하거나 정의합니다. |
| 174 | `    else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 175 | `        for local_key, global_coord_val in local_to_global_map.items():` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 176 | `            dist = np.linalg.norm(model_choice_coord - np.array(local_key))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 177 | `            ` | 가독성을 위해 구분한 빈 줄입니다. |
| 178 | `            if dist < min_dist:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 179 | `                min_dist = dist` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 180 | `                best_match_global_coord = global_coord_val` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 181 | `    target_position0 = best_match_global_coord` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 182 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 183 | `    return target_position0, is_final_decision,output_texts` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 184 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 185 | `def get_result_fast(qwen, processor,bank,global_color_list,global_list,sim,agent,target_position,log_file_path, current_frontiers, decision_agent_state, object_category, decision_num,rot` | 함수 `get_result_fast` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 186 | `               ):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 187 | `    def log_message(message):` | 함수 `log_message` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 188 | `        with open(log_file_path, "a") as f:` | 컨텍스트 매니저를 사용해 리소스를 안전하게 열고 자동 정리합니다. |
| 189 | `            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 190 | `            f.write(f"[{timestamp}] {message}\n") ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 191 | `    ref_coord = decision_agent_state.position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 192 | `    ref_quat_dict = {` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 193 | `        'x': decision_agent_state.rotation.x, 'y': decision_agent_state.rotation.y,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 194 | `        'z': decision_agent_state.rotation.z, 'w': decision_agent_state.rotation.w` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 195 | `    }` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 196 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 197 | `    input_pos = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 198 | `    all_prompt_images = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 199 | `    REPRODUCIBLE_SEED = 42` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 200 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 201 | `    all_fwd_indices_before_start = [i for i, act in enumerate(global_color_list) if act[1] == 'goto']` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 202 | `    if rot >100:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 203 | `        log_message('too many interrupt')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 204 | `        return '000',global_list[-1],rot` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 205 | `    if check_small_position_change(` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 206 | `        all_fwd_indices_before_start, ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 207 | `        global_list, ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 208 | `        threshold=0.1,  ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 209 | `        lookback=20` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 210 | `    ):  ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 211 | `        import habitat_sim` | 필요한 모듈 `habitat_sim` 를 현재 네임스페이스에 로드합니다. |
| 212 | `        agent_state = habitat_sim.AgentState()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 213 | `        rot=rot+1` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 214 | `        while True:` | 조건이 참인 동안 같은 블록을 반복 실행하는 루프입니다. |
| 215 | `            noise_scale=0.3` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 216 | `            dx = np.random.uniform(-noise_scale, noise_scale)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 217 | `            dy = np.random.uniform(-noise_scale, noise_scale)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 218 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 219 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 220 | `            new_pos=agent.state.position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 221 | `            new_pos[0]=new_pos[0]+dy` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 222 | `            new_pos[2]=new_pos[2]+dx` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 223 | `            new_pos=np.array(sim.step_filter(agent.state.position,new_pos ))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 224 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 225 | `            if sim.pathfinder.is_navigable(new_pos) and np.linalg.norm(new_pos-agent.state.position)>0.01:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 226 | `                break` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 227 | `        print('interrupt')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 228 | `        log_message('interrupt')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 229 | `        agent_state.position=new_pos` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 230 | `        agent_state.rotation = [decision_agent_state.rotation.x,decision_agent_state.rotation.y,decision_agent_state.rotation.z,decision_agent_state.rotation.w]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 231 | `        agent.set_state(agent_state)            ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 232 | `        return sim.get_sensor_observations(agent_ids=[0])[0],[transform_from_local_frame(np.insert(agent_state.position, 1, 0), ref_coord, ref_quat_dict)] ,rot` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 233 | `    range_start_idx = 0 ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 234 | `    if len(all_fwd_indices_before_start) >= 12:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 235 | `        range_start_idx = all_fwd_indices_before_start[-12]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 236 | `    range_end_idx = len(global_color_list)-2` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 237 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 238 | `    indices = np.linspace(range_start_idx, range_end_idx, 4, dtype=int).tolist()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 239 | `    random.seed(REPRODUCIBLE_SEED)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 240 | `    goto_image_select = [global_color_list[i][0] for i in indices]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 241 | `    goto_state_select = [global_list[i] for i in indices]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 242 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 243 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 244 | `    all_prompt_images.extend(goto_image_select)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 245 | `    obervations =  sim.get_sensor_observations(agent_ids=[0])[0]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 246 | `    all_prompt_images.extend([obervations['color_sensor_left'][:, :, :3]])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 247 | `    all_prompt_images.extend([obervations['color_sensor'][:, :, :3]])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 248 | `    all_prompt_images.extend([obervations['color_sensor_right'][:, :, :3]])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 249 | `    input_pos.extend(goto_state_select)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 250 | `    input_pos.extend([global_list[-1]])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 251 | `    input_pos.extend([target_position])   ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 252 | `    content_text ="The following are observation images from the past 4 frames:<image>,<image>,<image>,<image>\n\` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 253 | `        The current tri-view is shown below: leftside:<image>,frontside:<image>,rightside:<image>\n\` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 254 | `        Position coordinates for the past 4 frames:<input_pos1><input_pos2><input_pos3><input_pos4>\n\` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 255 | `        The current observation represents the coordinate: <input_pos5>\n\` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 256 | `        Target position coordinate: <input_target>\n\` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 257 | `        Please predict the position coordinates for the next 5 frames based on the above information.<\|NAV\|>\nOutput the waypoint"    # --- 3. 构建当前图像部分 <image> ---` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 258 | `    import numpy` | 필요한 모듈 `numpy` 를 현재 네임스페이스에 로드합니다. |
| 259 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 260 | `    input_pos_local=[]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 261 | `    for frontier_coord in input_pos:` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 262 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 263 | `        local_coord = transform_to_local_frame(frontier_coord.position if type(frontier_coord) != numpy.ndarray else frontier_coord, ref_coord, ref_quat_dict)` | 월드 좌표를 에이전트 기준 로컬 좌표계로 변환하는 유틸리티를 호출하거나 정의합니다. |
| 264 | `        x, z = local_coord[0], local_coord[2]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 265 | `        input_pos_local.append([x,z])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 266 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 267 | `    current_messages=[]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 268 | ` ` | 가독성을 위해 구분한 빈 줄입니다. |
| 269 | `    msg_copy={}` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 270 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 271 | `    if len(current_messages) == 0:  ` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 272 | `        content = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 273 | `        content.append({"type": "image", "image": all_prompt_images})` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 274 | `        content.append({"type": "text", "text":content_text})` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 275 | `        msg_copy['content'] = content` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 276 | `        current_messages.append(msg_copy)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 277 | `    current_messages[0]['role']='user'` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 278 | `    text = processor.apply_chat_template(` | Qwen 프로세서의 채팅 템플릿으로 멀티모달 입력을 모델이 이해하는 프롬프트 문자열로 변환합니다. |
| 279 | `        current_messages, tokenize=False, add_generation_prompt=True` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 280 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 281 | `    text=text.replace('<\|vision_start\|><\|image_pad\|><\|vision_end\|>','')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 282 | `    text=text.replace('<image>','<\|vision_start\|><\|image_pad\|><\|vision_end\|>')` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 283 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 284 | `    image_inputs, video_inputs = process_vision(all_prompt_images)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 285 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 286 | `    from transformers import AutoTokenizer` | `from transformers import AutoTokenizer` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 287 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 288 | `    image_inputs=preprocess(image_inputs)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 289 | `    inputs = processor(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 290 | `        text=text,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 291 | `        images=image_inputs,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 292 | `        padding=True,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 293 | `        padding_side="left",` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 294 | `        return_tensors="pt",` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 295 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 296 | `    device = qwen.device ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 297 | `    step_scale = 0.3` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 298 | `    input_positions = torch.tensor(input_pos_local,dtype=torch.float32)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 299 | `    input_positions_scaled = (input_positions / step_scale)[None].to(device)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 300 | `    inputs['input_waypoints'] = input_positions_scaled` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 301 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 302 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 303 | `    inputs = inputs.to(device)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 304 | `    # print(frontier_content)` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 305 | `    print(text.count('image_pad'))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 306 | `    with torch.no_grad():` | 컨텍스트 매니저를 사용해 리소스를 안전하게 열고 자동 정리합니다. |
| 307 | `        wp_pred, arrive_pred,sin_angle,cos_angle = qwen.forward(**inputs, action_former=True, gt_waypoints=0,train=False,train_branch=['continue'])` | 행동 예측 분기에서 모델 순전파를 호출해 waypoint/도착/각도 관련 출력을 얻습니다. |
| 308 | `    wp_pred = wp_pred.cpu().type(torch.float32).numpy().squeeze()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 309 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 310 | `    recover_angle = torch.atan2(sin_angle, cos_angle).detach().cpu().type(torch.float32).numpy().squeeze()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 311 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 312 | `    select_way_point_idx = 0` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 313 | `    way_point_loc = wp_pred[select_way_point_idx]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 314 | `    # way_point_coord= np.insert(way_point_loc, 1, 0)` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 315 | `    # pos=transform_from_local_frame(way_point_coord,ref_coord, ref_quat_dict) ` | 로컬 좌표를 다시 월드 좌표로 복원하는 유틸리티를 호출하거나 정의합니다. |
| 316 | `    r=np.linalg.norm(way_point_loc)*step_scale` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 317 | `    pos = rtheta_to_global_coordinates(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 318 | `        sim, agent,r, recover_angle[0], y_delta=0, dimensionality=3` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 319 | `    )    ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 320 | `    agent_pos = agent.get_state().position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 321 | `    new_rot = agent.get_state().rotation ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 322 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 323 | `    new_pos = np.array(sim.step_filter(agent_pos, pos))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 324 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 325 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 326 | `    if np.any(np.isnan(new_pos)) or not sim.pathfinder.is_navigable(new_pos):` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 327 | `        new_pos = agent_pos` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 328 | `        new_rot, _ = compute_heading_to(agent_pos, pos)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 329 | `    else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 330 | `        new_pos = np.array(sim.pathfinder.snap_point(new_pos))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 331 | `        if np.any(np.isnan(new_pos)) or not sim.pathfinder.is_navigable(new_pos):` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 332 | `            new_pos = agent_pos` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 333 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 334 | `        new_rot, _ = compute_heading_to(agent_pos, pos)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 335 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 336 | `    import habitat_sim` | 필요한 모듈 `habitat_sim` 를 현재 네임스페이스에 로드합니다. |
| 337 | `    agent_state = habitat_sim.AgentState()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 338 | `    agent_state.position = new_pos` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 339 | `    agent_state.rotation = new_rot` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 340 | `    agent.set_state(agent_state)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 341 | `    # obs = sim.get_observations()` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 342 | `    log_message(f"target_position:{input_pos_local[-1]},wp_pred:{wp_pred[0]},recover_angle:{recover_angle[0]}")` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 343 | `    return sim.get_sensor_observations(agent_ids=[0])[0],[transform_from_local_frame(np.insert(way_point_coord, 1, 0), ref_coord, ref_quat_dict) for  way_point_coord in wp_pred],rot` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 344 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 345 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 346 | `class Bank:` | 클래스 `Bank` 를 선언해 관련 상태와 메서드를 묶어 관리합니다. |
| 347 | `    def __init__(self, goto_sample_interval=3):` | 함수 `__init__` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 348 | `        self.images_spin = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 349 | `        self.agent_states_spin = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 350 | `        self.images_goto = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 351 | `        self.agent_states_goto = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 352 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 353 | `        self.unsampled_goto_images = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 354 | `        self.unsampled_goto_agent_states = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 355 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 356 | `        self.goto_sample_interval = goto_sample_interval` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 357 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 358 | `    def add(self, images, agent_states, data_type='spin'):` | 함수 `add` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 359 | `        if not images:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 360 | `            return` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 361 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 362 | `        if data_type == 'goto':` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 363 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 364 | `            sampled_indices = set(self._get_sampled_indices(len(images)))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 365 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 366 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 367 | `            for i, (img, state) in enumerate(zip(images, agent_states)):` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 368 | `                if i in sampled_indices:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 369 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 370 | `                    self.images_goto.append(img)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 371 | `                    self.agent_states_goto.append(state)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 372 | `                else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 373 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 374 | `                    self.unsampled_goto_images.append(img)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 375 | `                    self.unsampled_goto_agent_states.append(state)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 376 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 377 | `        else: ` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 378 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 379 | `            self.images_spin.extend(images)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 380 | `            self.agent_states_spin.extend(agent_states)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 381 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 382 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 383 | `    def get_spin_data(self):` | 함수 `get_spin_data` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 384 | `        return self.images_spin, self.agent_states_spin` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 385 | `    def get_goto_data(self):` | 함수 `get_goto_data` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 386 | `        return self.images_goto, self.agent_states_goto` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 387 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 388 | `    def get_unsampled_goto_data(self):` | 함수 `get_unsampled_goto_data` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 389 | `        return self.unsampled_goto_images, self.unsampled_goto_agent_states` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 390 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 391 | `    def clear(self):` | 함수 `clear` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 392 | `        self.images_spin.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 393 | `        self.agent_states_spin.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 394 | `        self.images_goto.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 395 | `        self.agent_states_goto.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 396 | `        self.unsampled_goto_images.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 397 | `        self.unsampled_goto_agent_states.clear()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 398 | `        print("Bank memory cleared.")` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 399 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 400 | `    def _get_sampled_indices(self, num_images):` | 함수 `_get_sampled_indices` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 401 | `        if self.goto_sample_interval==1:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 402 | `            return list(range(num_images))` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 403 | `        if num_images <= 2:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 404 | `            return list(range(num_images))` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 405 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 406 | `        sampled_indices = {0, num_images - 1}` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 407 | `        step = self.goto_sample_interval + 1` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 408 | `        for i in range(step, num_images - 1, step):` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 409 | `            sampled_indices.add(i)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 410 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 411 | `        return sorted(list(sampled_indices))` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 412 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 413 | `    def __len__(self):` | 함수 `__len__` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 414 | `        return len(self.images)` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 415 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 416 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 417 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 418 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 419 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 420 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 421 | `from habitat.utils.geometry_utils import (` | `from habitat.utils.geometry_utils import (` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 422 | `    quaternion_rotate_vector,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 423 | `    quaternion_to_list,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 424 | `)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 425 | `def compute_heading_to(` | 함수 `compute_heading_to` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 426 | `    pos_from, pos_to` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 427 | `):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 428 | `    """Compute the heading that points from position \`pos_from\` to position \`pos_to\`` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 429 | `    in the global XZ coordinate frame.` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 430 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 431 | `    Args:` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 432 | `        pos_from: [x,y,z] or [x,z]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 433 | `        pos_to: [x,y,z] or [x,z]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 434 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 435 | `    Returns:` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 436 | `        heading quaternion as [x, y, z, w]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 437 | `        heading scalar angle` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 438 | `    """` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 439 | `    delta_x = pos_to[0] - pos_from[0]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 440 | `    delta_z = pos_to[-1] - pos_from[-1]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 441 | `    xz_angle = np.arctan2(delta_x, delta_z)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 442 | `    xz_angle = (xz_angle + np.pi) % (2 * np.pi)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 443 | `    quat = quaternion_to_list(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 444 | `        quaternion.from_euler_angles([0.0, xz_angle, 0.0])` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 445 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 446 | `    return quat, xz_angle` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 447 | `from datetime import datetime` | `from datetime import datetime` 구문으로 특정 모듈/심볼을 직접 가져와 아래 코드에서 바로 사용 가능하게 만듭니다. |
| 448 | `import torch` | 필요한 모듈 `torch` 를 현재 네임스페이스에 로드합니다. |
| 449 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 450 | `def find_first_waypoint_beyond_threshold(wp_pred, threshold=0.15):` | 함수 `find_first_waypoint_beyond_threshold` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 451 | `    wp_pred = torch.from_numpy(wp_pred)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 452 | `    if wp_pred.dim() == 3:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 453 | `        # shape: [batch_size, num_waypoints, 2]` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 454 | `        batch_size = wp_pred.shape[0]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 455 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 456 | `        distances = torch.norm(wp_pred, dim=-1)  # [batch_size, num_waypoints]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 457 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 458 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 459 | `        indices = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 460 | `        for i in range(batch_size):` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 461 | `            dist = distances[i]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 462 | `            mask = dist > threshold` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 463 | `            ` | 가독성을 위해 구분한 빈 줄입니다. |
| 464 | `            if mask.any():` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 465 | `                idx = torch.nonzero(mask, as_tuple=True)[0][0].item()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 466 | `            else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 467 | `                idx = wp_pred.shape[1] - 1` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 468 | `            ` | 가독성을 위해 구분한 빈 줄입니다. |
| 469 | `            indices.append(idx)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 470 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 471 | `        indices = torch.tensor(indices, device=wp_pred.device)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 472 | `        return indices` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 473 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 474 | `    elif wp_pred.dim() == 2:` | 이전 `if` 조건이 거짓일 때 검사되는 추가 조건 분기입니다. |
| 475 | `        # shape: [num_waypoints, 2]` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 476 | `        distances = torch.norm(wp_pred, dim=-1)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 477 | `        mask = distances > threshold` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 478 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 479 | `        if mask.any():` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 480 | `            idx = torch.nonzero(mask, as_tuple=True)[0][0].item()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 481 | `        else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 482 | `            idx = wp_pred.shape[0] - 1` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 483 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 484 | `        return idx` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 485 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 486 | `    else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 487 | `        raise ValueError(f"Unexpected wp_pred shape: {wp_pred.shape}")` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 488 | `import math` | 필요한 모듈 `math` 를 현재 네임스페이스에 로드합니다. |
| 489 | `import habitat_sim` | 필요한 모듈 `habitat_sim` 를 현재 네임스페이스에 로드합니다. |
| 490 | `def rtheta_to_global_coordinates(` | 함수 `rtheta_to_global_coordinates` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 491 | `    sim,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 492 | `    agent,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 493 | `    r,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 494 | `    theta,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 495 | `    y_delta,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 496 | `    dimensionality,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 497 | `):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 498 | `    """Maps relative polar coordinates from an agent position to an updated` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 499 | `    agent position. The returned position is not validated for navigability.` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 500 | `    """` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 501 | `    assert dimensionality in [2, 3]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 502 | `    scene_node = sim.get_agent(0).scene_node` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 503 | `    forward_ax = (` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 504 | `        np.array(scene_node.absolute_transformation().rotation_scaling())` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 505 | `        @ habitat_sim.geo.FRONT` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 506 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 507 | `    agent_state = agent.get_state()` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 508 | `    rotation = habitat_sim.utils.quat_from_angle_axis(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 509 | `        theta, habitat_sim.geo.UP` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 510 | `    )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 511 | `    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 512 | `    position = agent_state.position + (move_ax * r)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 513 | `    position[1] += y_delta` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 514 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 515 | `    if dimensionality == 2:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 516 | `        return [position[0], position[2]]` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 517 | `    return position` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 518 | `import numpy as np` | 필요한 모듈 `numpy as np` 를 현재 네임스페이스에 로드합니다. |
| 519 | `def rescale_image_with_long_edge_and_random_scale(image: Image.Image, ` | 함수 `rescale_image_with_long_edge_and_random_scale` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 520 | `                                                 long_edge=640, ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 521 | `                                                 scale=1.0):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 522 | `    w, h = image.size` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 523 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 524 | `    if w >= h:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 525 | `        new_w = long_edge` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 526 | `        new_h = int(h * (long_edge / w))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 527 | `    else:` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 528 | `        new_h = long_edge` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 529 | `        new_w = int(w * (long_edge / h))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 530 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 531 | `    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 532 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 533 | `    final_w = int(new_w * scale)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 534 | `    final_h = int(new_h * scale)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 535 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 536 | `    if final_w < 1 or final_h < 1:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 537 | `        final_w = max(1, final_w)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 538 | `        final_h = max(1, final_h)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 539 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 540 | `    image = image.resize((final_w, final_h), Image.Resampling.LANCZOS)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 541 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 542 | `    return image` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 543 | `import random` | 필요한 모듈 `random` 를 현재 네임스페이스에 로드합니다. |
| 544 | `import math` | 필요한 모듈 `math` 를 현재 네임스페이스에 로드합니다. |
| 545 | `def rescale_image_to_fixed_size(img: Image.Image, height: int, width: int) -> Image.Image:` | 함수 `rescale_image_to_fixed_size` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 546 | `    import torchvision.transforms as T` | 필요한 모듈 `torchvision.transforms as T` 를 현재 네임스페이스에 로드합니다. |
| 547 | `    return T.Resize((int(height), int(width)))(img)` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 548 | `def crop_resize_image_magic_resolution(img: Image.Image) -> Image.Image:` | 함수 `crop_resize_image_magic_resolution` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 549 | `    import torchvision.transforms as T` | 필요한 모듈 `torchvision.transforms as T` 를 현재 네임스페이스에 로드합니다. |
| 550 | `    width = img.width` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 551 | `    height = img.height` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 552 | `    if width !=720 and height !=640:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 553 | `        return img` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 554 | `    top, bottom = 140, 500` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 555 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 556 | `    left, right = 0, img.width` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 557 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 558 | `    cropped_img = img.crop((left, top, right, bottom))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 559 | `    resized_img = cropped_img.resize((width, height), Image.BILINEAR)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 560 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 561 | `    return resized_img` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 562 | `def round_by_factor(number: int, factor: int) -> int:` | 함수 `round_by_factor` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 563 | `    """Returns the closest integer to 'number' that is divisible by 'factor'."""` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 564 | `    return round(number / factor) * factor` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 565 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 566 | `def floor_by_factor(number: int, factor: int) -> int:` | 함수 `floor_by_factor` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 567 | `    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 568 | `    return math.floor(number / factor) * factor` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 569 | `def ceil_by_factor(number: int, factor: int) -> int:` | 함수 `ceil_by_factor` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 570 | `    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 571 | `    return math.ceil(number / factor) * factor` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 572 | `def smart_resize(` | 함수 `smart_resize` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 573 | `    height, width, factor, min_pixels, max_pixels` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 574 | `):` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 575 | `    """` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 576 | `    Rescales the image so that the following conditions are met:` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 577 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 578 | `    1. Both dimensions (height and width) are divisible by 'factor'.` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 579 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 580 | `    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 581 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 582 | `    3. The aspect ratio of the image is maintained as closely as possible.` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 583 | `    """` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 584 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 585 | `    h_bar = max(factor, round_by_factor(height, factor))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 586 | `    w_bar = max(factor, round_by_factor(width, factor))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 587 | `    if h_bar * w_bar > max_pixels:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 588 | `        beta = math.sqrt((height * width) / max_pixels)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 589 | `        h_bar = floor_by_factor(height / beta, factor)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 590 | `        w_bar = floor_by_factor(width / beta, factor)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 591 | `    elif h_bar * w_bar < min_pixels:` | 이전 `if` 조건이 거짓일 때 검사되는 추가 조건 분기입니다. |
| 592 | `        beta = math.sqrt(min_pixels / (height * width))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 593 | `        h_bar = ceil_by_factor(height * beta, factor)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 594 | `        w_bar = ceil_by_factor(width * beta, factor)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 595 | `    return h_bar, w_bar` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 596 | `def preprocess(images):` | 함수 `preprocess` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 597 | `    images = [` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 598 | `        crop_resize_image_magic_resolution(img) ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 599 | `            if idx not in {len(images) - 1, len(images) - 3} ` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 600 | `            else img ` | 앞선 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. |
| 601 | `            for idx, img in enumerate(images)` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 602 | `        ]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 603 | `    current_img_num = 3` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 604 | `    LONG_EDGE = 640` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 605 | `    SCALE_RANGE = (0.7, 1.0)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 606 | `    # ✅ 在 batch 级别随机选择一个 scale（所有图像共用）` | 주석으로 코드 의도/형상 정보를 설명합니다. |
| 607 | `    s = 1.0` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 608 | `    images = [` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 609 | `        rescale_image_with_long_edge_and_random_scale(img, LONG_EDGE, scale = s)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 610 | `        for img in images` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 611 | `    ]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 612 | `    if True:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 613 | `        images = [rescale_image_to_fixed_size(img,int(img.height/4),int(img.width/4)) if idx < len(images)-current_img_num else img for idx,img in enumerate(images)]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 614 | `    images_new  = []` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 615 | `    size_factor = 28 ` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 616 | `    min_pixels = 3136` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 617 | `    max_pixels = 12845056` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 618 | `    for image in images:` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 619 | `        width, height = image.size` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 620 | `        resized_height, resized_width = smart_resize(` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 621 | `            height,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 622 | `            width,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 623 | `            factor=size_factor,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 624 | `            min_pixels=min_pixels,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 625 | `            max_pixels=max_pixels,` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 626 | `        )` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 627 | `        image = image.resize((resized_width, resized_height))` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 628 | `        images_new.append(image)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 629 | `    return   images_new             ` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 630 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 631 | `def check_small_position_change(all_fwd_indices_before_start, positions, threshold=0.2, lookback=50):` | 함수 `check_small_position_change` 를 선언하는 시작 줄이며, 아래 들여쓰기 블록이 실제 동작을 정의합니다. |
| 632 | `    if len(all_fwd_indices_before_start) < lookback:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 633 | `        return False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 634 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 635 | `    recent_indices = all_fwd_indices_before_start[-lookback:]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 636 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 637 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 638 | `    for i in range(len(recent_indices) - 1):` | 반복문 시작으로, 시퀀스 원소를 순회하며 블록을 반복 실행합니다. |
| 639 | `        idx1 = recent_indices[i]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 640 | `        idx2 = recent_indices[i + 1]` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 641 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 642 | `        pos1 = positions[idx1].position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 643 | `        pos1[1]=0` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 644 | `        pos2 = positions[idx2].position` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 645 | `        pos2[1]=0` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 646 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 647 | `        distance = np.linalg.norm(pos2 - pos1)` | 해당 줄은 변수 설정·연산·데이터 조립 등 현재 로직의 세부 동작을 수행합니다. |
| 648 | `        ` | 가독성을 위해 구분한 빈 줄입니다. |
| 649 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 650 | `        if distance > threshold:` | 조건문 분기 시작으로, 조건이 참일 때만 다음 블록을 실행합니다. |
| 651 | `            return False` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 652 | `    ` | 가독성을 위해 구분한 빈 줄입니다. |
| 653 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 654 | `    return True` | 함수 실행을 종료하고 호출자에게 값을 반환합니다. |
| 655 | `` | 가독성을 위해 구분한 빈 줄입니다. |
| 656 | `import torch` | 필요한 모듈 `torch` 를 현재 네임스페이스에 로드합니다. |


## 핵심 로직 흐름 추가 해설

아래는 위 라인별 표를 읽기 쉽게 보완하기 위한 함수/클래스 단위 설명입니다.
- `parse_response` (`14~31`): 모델 자유형 텍스트에서 `coordinate [...]` 패턴을 추출하고 JSON 파싱 후 found 여부를 불리언으로 반환합니다.
- `process_vision` (`33~44`): numpy 이미지 배열 리스트를 PIL 이미지로 변환 후 고정 해상도(486x420)로 맞춰 모델 입력을 균질화합니다.
- `transform_to_local_frame` (`50~59`): 월드 기준 점을 에이전트 기준 로컬 좌표로 변환하고 축 부호를 프로젝트 규약에 맞게 뒤집습니다.
- `transform_from_local_frame` (`61~74`): 로컬 좌표를 월드 좌표로 되돌리는 역변환이며 경로점 복원에 사용됩니다.
- `getresult` (`96~183`): 파노라마/레퍼런스/프론티어 좌표를 프롬프트로 구성해 Qwen 생성 결과를 좌표로 파싱하고 최종 목표점을 결정합니다.
- `get_result_fast` (`185~343`): 과거 프레임+현재 삼안 이미지를 입력해 waypoint를 예측하고 시뮬레이터 agent 상태를 갱신합니다.
- `Bank` (`346~414`): 스핀/고투 이미지와 상태를 분리 저장하고 샘플링하는 메모리 뱅크 역할을 수행합니다.
- `compute_heading_to` (`425~446`): 현재 위치에서 목표 위치를 바라보는 yaw 회전을 quaternion과 각도로 계산합니다.
- `find_first_waypoint_beyond_threshold` (`450~487`): 예측 waypoint들 중 원점에서 일정 거리 이상 떨어진 첫 인덱스를 선택합니다.
- `rtheta_to_global_coordinates` (`490~517`): 상대 극좌표(r,theta)를 월드 좌표로 사상하여 다음 이동 후보점을 생성합니다.
- `rescale_image_with_long_edge_and_random_scale` (`519~542`): 긴 변 기준 리사이즈 후 추가 스케일링을 적용합니다.
- `rescale_image_to_fixed_size` (`545~547`): 지정 높이/너비로 강제 리사이즈합니다.
- `crop_resize_image_magic_resolution` (`548~561`): 특정 입력 해상도 조건에서만 중앙 영역을 크롭 후 원래 해상도로 보간합니다.
- `round_by_factor` (`562~564`): 지정 factor 배수로 반올림합니다.
- `floor_by_factor` (`566~568`): 지정 factor 배수로 내림합니다.
- `ceil_by_factor` (`569~571`): 지정 factor 배수로 올림합니다.
- `smart_resize` (`572~595`): 해상도를 factor 배수/최소·최대 픽셀 조건을 동시에 만족하도록 계산합니다.
- `preprocess` (`596~629`): 크롭·스케일·축소·smart_resize를 순차 적용해 멀티뷰 입력 이미지를 모델 친화적으로 전처리합니다.
- `check_small_position_change` (`631~654`): 최근 이동 이력이 거의 정지 상태인지 거리 임계값으로 판정해 인터럽트 로직 트리거에 사용합니다.
