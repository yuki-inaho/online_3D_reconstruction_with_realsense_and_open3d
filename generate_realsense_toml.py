from collections import OrderedDict
from scripts.realsense import RealSenseManager
from pathlib import Path
import toml

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
CFG_DIR_PATH = Path(SCRIPT_DIR_PATH, "cfg")


def set_intrinsics(dict_toml, key_name, intrinsic):
    dict_toml[key_name]["fx"] = intrinsic.fx
    dict_toml[key_name]["fy"] = intrinsic.fy
    dict_toml[key_name]["cx"] = intrinsic.cx
    dict_toml[key_name]["cy"] = intrinsic.cy


def set_translation(dict_toml, key_name, translation):
    for i, elem in enumerate(["tx", "ty", "tz"]):
        dict_toml[key_name][elem] = float(translation[i])


def set_rotation(dict_toml, key_name, rotation_dcm):
    rotation_dcm_flattened = rotation_dcm.flatten()
    for i, elem in enumerate(["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"]):
        dict_toml[key_name][elem] = float(rotation_dcm_flattened[i])


def main():
    cfg_template_path = str(Path(CFG_DIR_PATH, "realsense_toml_template.toml"))
    cfg_output_path = str(Path(CFG_DIR_PATH, "realsense.toml"))

    decoder = toml.TomlDecoder(_dict=OrderedDict)
    encoder = toml.TomlEncoder(_dict=OrderedDict)
    toml.TomlEncoder = encoder
    dict_toml = toml.load(open(cfg_template_path), _dict=OrderedDict, decoder=decoder)

    rs_mng = RealSenseManager()
    intrinsic_depth = rs_mng.intrinsic_depth
    intrinsic_color = rs_mng.intrinsic_color
    intrinsic_ir_left = rs_mng.intrinsic_ir_left
    intrinsic_ir_right = rs_mng.intrinsic_ir_right

    translation_l2r = rs_mng.translation_ir_left2right
    translation_l2c = rs_mng.translation_ir_left2color
    rotation_dcm_l2r = rs_mng.rotation_dcm_ir_left2right
    rotation_dcm_l2c = rs_mng.rotation_dcm_ir_left2color


    set_intrinsics(dict_toml, "RGB_Intrinsics", intrinsic_color)
    set_intrinsics(dict_toml, "Depth_Intrinsics", intrinsic_depth)
    set_intrinsics(dict_toml, "IR_Left_Intrinsics", intrinsic_ir_left)
    set_intrinsics(dict_toml, "IR_Right_Intrinsics", intrinsic_ir_right)

    set_translation(dict_toml, "IR_L2R_Translation", translation_l2r)
    set_translation(dict_toml, "IR_L2C_Translation",translation_l2c)

    set_rotation(dict_toml, "IR_L2R_Rotation", rotation_dcm_l2r)
    set_rotation(dict_toml, "IR_L2C_Rotation", rotation_dcm_l2c)

    with open(cfg_output_path, "w") as f:
        toml.encoder.dump(dict_toml, f)
        print("generated")

    del rs_mng

if __name__ == "__main__":
    main()