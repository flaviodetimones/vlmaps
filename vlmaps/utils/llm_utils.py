import os
import openai


def parse_object_goal_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    question = f"""
    I: go to the kitchen and then go to the toilet. A: kitchen, toilet
    I: go to the chair and then go to another chair. A: chair, chair
    I: navigate to the green sofa and turn right and find several chairs, finally go to the painting. A: green sofa, chairs, painting
    I: approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen. A: window, television, oven, kitchen
    I: walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed. A: plant, table, bedroom, bed
    I: go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room. A: stairs, room, book shelf, table, next room
    I: Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances. A: table, cushion, column, appliances.
    I: Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table. A: chair, table
    I: {language_instr}. A:"""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"].strip()
    print("landmarks: ", result)
    return [x.strip() for x in result.split(",")]

def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of navigable semantic object categories.
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    Example: "go to the area of the counter" -> ["counter"]
    """
    import json
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    client = openai.OpenAI(api_key=openai_key)

    system_prompt = (
        "You extract navigable semantic object categories from robot navigation instructions.\n"
        "\n"
        "Rules:\n"
        "1. Output ONLY a JSON array of strings, nothing else. No explanation, no punctuation outside the array.\n"
        "2. Each string must be a physical object or place a robot can navigate to (e.g. 'counter', 'sofa', 'kitchen').\n"
        "2b. Preserve explicit room-instance identifiers when the user gives them "
        "(e.g. 'bathroom 1', 'bedroom.001'). Do not collapse them to the base room name.\n"
        "3. IGNORE spatial/relational words that are NOT objects themselves:\n"
        "   area, zone, region, vicinity, neighborhood, side, part, spot,\n"
        "   near, nearby, next to, beside, between, around, in front of, behind,\n"
        "   left, right, north, south, east, west, middle, center.\n"
        "   When these words modify an object (e.g. 'area of the counter'), extract only the object ('counter').\n"
        "4. Keep meaningful color/size/material adjectives that disambiguate objects (e.g. 'green sofa').\n"
        "5. Extract objects in the order they should be visited.\n"
        "6. If the same object appears multiple times, repeat it.\n"
        "7. Do not add objects not mentioned in the instruction."
    )

    few_shots = [
        # basic multi-target
        ("go to the kitchen and then go to the toilet",
         '["kitchen", "toilet"]'),
        ("go to bathroom 1",
         '["bathroom 1"]'),
        ("go to bedroom.001",
         '["bedroom.001"]'),
        ("go to the chair and then go to another chair",
         '["chair", "chair"]'),
        ("navigate to the green sofa and turn right and find several chairs, finally go to the painting",
         '["green sofa", "chairs", "painting"]'),
        ("approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen",
         '["window", "television", "oven", "kitchen"]'),
        ("walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed",
         '["plant", "table", "bedroom", "bed"]'),
        # directional noise — direction words are not targets
        ("Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degrees",
         '["chair", "sofa", "table"]'),
        # ── tricky relational/spatial qualifiers ──────────────────────────────
        # "area of X" → only X
        ("go to the area of the counter",
         '["counter"]'),
        ("navigate to the area near the bookshelf",
         '["bookshelf"]'),
        # "zone / region" → only the anchor object
        ("move to the zone around the refrigerator",
         '["refrigerator"]'),
        ("head to the region of the dining table",
         '["dining table"]'),
        # "between X and Y" → both objects
        ("go between the sofa and the TV",
         '["sofa", "TV"]'),
        ("position yourself between the bed and the wardrobe",
         '["bed", "wardrobe"]'),
        # "near / next to X" → only X
        ("stand near the window",
         '["window"]'),
        ("move next to the kitchen counter",
         '["kitchen counter"]'),
    ]

    messages = [{"role": "system", "content": system_prompt}]
    for user_text, assistant_text in few_shots:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": language_instr})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()

    # Robust parsing: try JSON first, fall back to comma-split
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: strip any accidental markdown fences, then split on commas
    text = text.strip("`").replace("```json", "").replace("```", "").strip()
    return [x.strip().strip('"').strip("'") for x in text.split(",") if x.strip()]



def parse_spatial_instruction_deprecated(language_instr):
    """
    [deprecated]: only for older version of OpenAI API
    """
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        question = f"""
# move a bit to the right of the refrigerator.
robot.move_to_right('refrigerator')
###
# move in between the couch and bookshelf.
robot.move_in_between('couch', 'bookshelf')
###
# move in the middle of the stairs and oven.
robot.move_in_between('stairs', 'oven')
###
# face the toilet.
robot.face('toilet')
###
# move to the south side of the table
robot.move_south('table')
###
# move to the west of the chair
robot.move_west('chair')
###
# move to the east of the chair
robot.move_east('chair')
###
# turn left 60 degrees
robot.turn(-60)
###
# turn right 20 degrees
robot.turn(20)
###
# find any chairs in the environment
robot.move_to_object('chair')
###
# come to the table
robot.move_to_object('table')
###
# with the television on your left
robot.with_object_on_left('television')
###
# with the toilet to your right, turn left 30 degrees
robot.with_object_on_right('toilet')
robot.turn(-30)
###
# with the toilet to your left, turn right 30 degrees
robot.with_object_on_left('toilet')
robot.turn(30)
###
# with the television behind you
robot.face('television')
robot.turn(180)
###
# go to the south of the table in front of you, with the table on your left, turn right 45 degrees
robot.move_south('table')
robot.with_object_on_left('table')
robot.turn(45)
###
# move forward for 3 meters
robot.move_forward(3)
###
# move right 2 meters
robot.turn(90)
robot.move_forward(2)
###
# move left 3 meters
robot.turn(-90)
robot.move_forward(3)
###
# move back and forth between the chair and the table 3 times
pos1 = robot.get_pos('chair')
pos2 = robot.get_pos('table')
for i in range(3):
    robot.move_to(pos1)
    robot.move_to(pos2)
###
# go to the television
robot.move_to_object('television')
###
# monitor the chair from 4 view points
contour = robot.get_contour('chair')
np.random.shuffle(contour)
for pos in contour[:4]:
    robot.move_to(pos)
    robot.face('chair')
###
# navigate to 3 meters right of the table
robot.move_to_right('table')
robot.face('table')
robot.turn(180)
robot.move_forward(3)
###
# turn west
robot.turn_absolute(-90)
###
# turn east
robot.turn_absolute(90)
###
# turn south
robot.turn_absolute(180)
###
# turn north
robot.turn_absolute(0)
###
# turn east and then turn left 90 degrees
robot.turn_absolute(90)
robot.turn(-90)
###
# move 3 meters north of the chair
robot.move_north('chair')
robot.turn_absolute(0)
robot.move_forward(3)
###
# move 1 meters south of the chair
robot.move_south('chair')
robot.turn_absolute(180)
robot.move_forward(3)
###
# move 3 meters west
robot.turn_absolute(-90)
robot.move_forward(3)
# {lang}
    """
        print("lang: ", lang)
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=question,
            max_tokens=400,
            temperature=0.0,
            stop="###",
        )
        result = response["choices"][0]["text"].strip()
        if result:
            results += result + "\n"
        print(result)
    print("generated code")
    print(results)
    return results

def parse_spatial_instruction(language_instr):
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "move a bit to the right of the refrigerator"},
                {"role": "assistant", "content": "robot.move_to_right('refrigerator')"},
                {"role": "user", "content": "move in between the couch and bookshelf"},
                {"role": "assistant", "content": "robot.move_in_between('couch', 'bookshelf')"},
                {"role": "user", "content": "move in the middle of the stairs and oven"},
                {"role": "assistant", "content": "robot.move_in_between('stairs', 'oven')"},
                {"role": "user", "content": "face the toilet"},
                {"role": "assistant", "content": "robot.face('toilet')"},
                {"role": "user", "content": "move to the south side of the table"},
                {"role": "assistant", "content": "robot.move_south('table')"},
                {"role": "user", "content": "move to the west of the chair"},
                {"role": "assistant", "content": "robot.move_west('chair')"},
                {"role": "user", "content": "move to the east of the chair"},
                {"role": "assistant", "content": "robot.move_east('chair')"},
                {"role": "user", "content": "turn left 60 degrees"},
                {"role": "assistant", "content": "robot.turn(-60)"},
                {"role": "user", "content": "turn right 20 degrees"},
                {"role": "assistant", "content": "robot.turn(20)"},
                {"role": "user", "content": "find any chairs in the environment"},
                {"role": "assistant", "content": "robot.move_to_object('chair')"},
                {"role": "user", "content": "come to the table"},
                {"role": "assistant", "content": "robot.move_to_object('table')"},
                {"role": "user", "content": "with the television on your left"},
                {"role": "assistant", "content": "robot.with_object_on_left('television')"},
                {"role": "user", "content": "with the toilet to your right, turn left 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_right('toilet')\nrobot.turn(-30)"},
                {"role": "user", "content": "with the toilet to your left, turn right 30 degrees"},
                {"role": "assistant", "content": "robot.with_object_on_left('toilet')\nrobot.turn(30)"},
                {"role": "user", "content": "with the television behind you"},
                {"role": "assistant", "content": "robot.face('television')\nrobot.turn(180)"},
                {"role": "user", "content": "go to the south of the table in front of you, with the table on your left, turn right 45 degrees"},
                {"role": "assistant", "content": "robot.move_south('table')\nrobot.with_object_on_left('table')\nrobot.turn(45)"},
                {"role": "user", "content": "move forward for 3 meters"},
                {"role": "assistant", "content": "robot.move_forward(3)"},
                {"role": "user", "content": "move right 2 meters"},
                {"role": "assistant", "content": "robot.turn(90)\nrobot.move_forward(2)"},
                {"role": "user", "content": "move left 3 meters"},
                {"role": "assistant", "content": "robot.turn(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": "move back and forth between the chair and the table 3 times"},
                {"role": "assistant", "content": "pos1 = robot.get_pos('chair')\npos2 = robot.get_pos('table')\nfor i in range(3):\n    robot.move_to(pos1)\n    robot.move_to(pos2)"},
                {"role": "user", "content": "go to the television"},
                {"role": "assistant", "content": "robot.move_to_object('television')"},
                {"role": "user", "content": "monitor the chair from 4 view points"},
                {"role": "assistant", "content": "contour = robot.get_contour('chair')\nnp.random.shuffle(contour)\nfor pos in contour[:4]:\n    robot.move_to(pos)\n    robot.face('chair')"},
                {"role": "user", "content": "navigate to 3 meters right of the table"},
                {"role": "assistant", "content": "robot.move_to_right('table')\nrobot.face('table')\nrobot.turn(180)\nrobot.move_forward(3)"},
                {"role": "user", "content": "turn west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)"},
                {"role": "user", "content": "turn east"},
                {"role": "assistant", "content": "robot.turn_absolute(90)"},
                {"role": "user", "content": "turn south"},
                {"role": "assistant", "content": "robot.turn_absolute(180)"},
                {"role": "user", "content": "turn north"},
                {"role": "assistant", "content": "robot.turn_absolute(0)"},
                {"role": "user", "content": "turn east and then turn left 90 degrees"},
                {"role": "assistant", "content": "robot.turn_absolute(90)\nrobot.turn(-90)"},
                {"role": "user", "content": "move 3 meters north of the chair"},
                {"role": "assistant", "content": "robot.move_north('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 1 meters south of the chair"},
                {"role": "assistant", "content": "robot.move_south('chair')\nrobot.move_forward(3)"},
                {"role": "user", "content": "move 3 meters west"},
                {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
                {"role": "user", "content": lang},
            ],
            max_tokens=300,
        )
        text = response.choices[0].message.content
        if text:
            results += text + "\n"
    return text
if __name__ == '__main__':
    text = parse_spatial_instruction("go to the sofa, turn right and move in between the table and the chair, and then move back and forth to the keyboard and the screen twice")
    print(text)
