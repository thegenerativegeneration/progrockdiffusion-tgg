import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--settings',
    required=False,
    default=['blending.json'],
    help='A settings JSON file to use, best to put in quotes.'
)

parser.add_argument("-x", '--xDimension', action='store_true',
                    help="Batch every possible setting in multidimensionnal grid.")

args = parser.parse_args()

# open the json file from the command line
try:
    with open(args.settings) as json_file:
        data = json.load(json_file)

except FileNotFoundError:
    print("File not found")
    exit()


def ConvertBooleanToFloat(value):
    if value:
        return 1.0
    else:
        return 0.0


# Lerp function
def lerp(a, b, t):
    booltype = False
    if type(a) == bool and type(b) == bool:
        a = ConvertBooleanToFloat(a)
        b = ConvertBooleanToFloat(b)
        booltype = True

    result = (1. - t) * a + t * b

    if type(a) == int:
        result = int(round(result))
    elif booltype:
        if round(result) > 0.5:
            return True
        else:
            return False

    return result


# Lerp function for dictionaries
def LerpDictionary(dict1, dict2, t):
    result = {}
    for key in dict1.keys():
        if key in dict2.keys():
            if type(dict1[key]) == dict:
                result[key] = LerpDictionary(dict1[key], dict2[key], t)
            else:
                result[key] = lerp(dict1[key], dict2[key], t)
        else:
            result[key] = dict1[key]
    return result


def ExecuteProgrockdiffusionWith(temporarySetting):
    # write a temporary setting file to override the base setting
    print("\nCurrent Setting:" + temporarySetting.__str__() + "\n\n")
    with open('tmp.json', 'w', encoding='utf-8') as f:
        json.dump(temporarySetting, f, ensure_ascii=False, indent=4)
    f.close()
    temporarySetting.clear()
    # Execute progrockdiffusion with the temporary setting file
    os.system('python prd.py -s settings.json -s tmp.json')


def N_DimensionalSetting():
    jsonData = []
    for dimension in range(len(data["settings"])):
        steps = data['settings'][dimension]['interpolation_steps']
        step_size = 1.0 / (steps - 1)
        jsonData.append([])
        for i in range(steps):
            current_step = i * step_size
            jsonData[dimension].append(
                LerpDictionary(data['settings'][dimension]['start'], data['settings'][dimension]['end'], current_step))

    meshgridSettings = np.array(np.meshgrid(*jsonData)).T.reshape(-1, len(data["settings"]))
    print("\n\Settings to be performed:" + meshgridSettings.__str__())

    for i in range(0, len(meshgridSettings)):
        setting = meshgridSettings[i][0]
        for j in range(1, len(meshgridSettings[i])):
            setting = {**setting, **meshgridSettings[i][j]}
        ExecuteProgrockdiffusionWith(setting)


def SequenceSettings():
    interpolatedSetting = {}
    for dimension in range(len(data["settings"])):
        # get the settings
        setting = data["settings"][dimension]
        steps = setting["interpolation_steps"]

        for step in range(steps):
            step_size = 1.0 / (steps - 1)
            current_step = step * step_size
            # Add the interpolated setting
            for key in setting["start"].keys():
                start = setting["start"][key]
                end = setting["end"][key]
                current_value = lerp(start, end, current_step)
                interpolatedSetting[key] = current_value
            ExecuteProgrockdiffusionWith(interpolatedSetting)


if args.xDimension:
    print("\n Batch every possible setting in multidimensionnal grid. \n")
    N_DimensionalSetting()
else:
    print("\n Batch settings in sequence.\n")
    SequenceSettings()

quit()
