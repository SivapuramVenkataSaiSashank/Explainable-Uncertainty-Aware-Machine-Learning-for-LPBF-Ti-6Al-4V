"""
build_dataset.py
Extracts and builds the standardized Process-Structure-Property (PSP) dataset
for LPBF Ti-6Al-4V from experimental tables and raw micro-CT CSV files.
"""

import os
import glob
import re
import pandas as pd
import numpy as np

def build_tables():
    # Table 2: Process Parameters and Descriptors (42 conditions)
    # Hatch spacing h = 82 um = 0.082 mm, Layer thickness t = 60 um = 0.060 mm
    t2_data = [
        (1, 275, 800, 344, 307, 220),
        (2, 275, 760, 362, 315, 209),
        (3, 275, 720, 382, 324, 198),
        (4, 275, 680, 404, 333, 187),
        (5, 275, 640, 430, 344, 176),
        (6, 275, 600, 458, 355, 165),
        (7, 275, 840, 327, 300, 231),
        (8, 275, 880, 313, 293, 242),
        (9, 275, 920, 299, 287, 253),
        (10, 275, 960, 286, 281, 264),
        (11, 275, 1000, 275, 275, 275),
        (12, 175, 800, 219, 196, 140),
        (13, 195, 800, 244, 218, 156),
        (14, 215, 800, 269, 240, 172),
        (15, 235, 800, 294, 263, 188),
        (16, 255, 800, 319, 285, 204),
        (17, 295, 800, 369, 330, 236),
        (18, 315, 800, 394, 352, 252),
        (19, 335, 800, 419, 375, 268),
        (20, 355, 800, 444, 397, 284),
        (21, 375, 800, 469, 419, 300),
        (22, 135, 400, 338, 213, 54),
        (23, 205, 600, 342, 265, 123),
        (24, 345, 1000, 345, 345, 345),
        (25, 415, 1200, 346, 379, 498),
        (26, 485, 1400, 346, 410, 679),
        (27, 500, 1480, 338, 411, 740),
        (28, 155, 400, 388, 245, 62),
        (29, 235, 600, 392, 303, 141),
        (30, 395, 1000, 395, 395, 395),
        (31, 475, 1200, 396, 434, 570),
        (32, 500, 1290, 388, 440, 645),
        (33, 115, 400, 288, 182, 46),
        (34, 175, 600, 292, 226, 105),
        (35, 295, 1000, 295, 295, 295),
        (36, 355, 1200, 296, 324, 426),
        (37, 475, 1600, 297, 376, 760),
        (38, 75, 400, 188, 119, 30),
        (39, 115, 600, 192, 148, 69),
        (40, 195, 1000, 195, 195, 195),
        (41, 235, 1200, 196, 215, 282),
        (42, 315, 1600, 197, 249, 504)
    ]
    df_proc = pd.DataFrame(t2_data, columns=[
        'Condition_ID', 'Laser_Power_W', 'Scan_Speed_mm_s', 'LED_J_m', 'MED_J_sqrt_m_s', 'Pv_W_m_s'
    ])
    # Add Volumetric Energy Density (VED) in J/mm^3: VED = P / (v * h * t)
    # v in mm/s, h = 0.082 mm, t = 0.060 mm
    h_mm = 0.082
    t_mm = 0.060
    df_proc['VED_J_mm3'] = (df_proc['Laser_Power_W'] / (df_proc['Scan_Speed_mm_s'] * h_mm * t_mm)).round(2)

    # Table 3: Porosity (35 conditions)
    t3_data = [
        (1, 0.8774, 0.0452, 1.3189, 0.8816),
        (2, 1.9382, 0.1495, 1.8791, 0.8775),
        (3, 2.3027, 0.5489, 2.1131, 0.8748),
        (4, 1.6876, 0.1638, 2.3318, 0.8728),
        (5, 2.0607, 0.0928, 2.5762, 0.8700),
        (6, 1.7519, 0.1885, 2.7962, 0.8677),
        (7, 1.2318, 0.0008, 1.2031, 0.8836),
        (8, 1.1349, 0.0489, 1.0150, 0.8853),
        (9, 0.5741, 0.0222, 0.8406, 0.8877),
        (10, 0.5011, 0.0508, 0.7168, 0.8880),
        (11, 0.2838, 0.0003, 0.6633, 0.8878),
        (12, 0.7368, 0.1135, 1.0992, 0.8835),
        (13, 0.5611, 0.1192, 1.1168, 0.8839),
        (14, 0.4444, 0.0017, 0.9152, 0.8853),
        (15, 0.7764, 0.1780, 1.1020, 0.8813),
        (16, 0.6089, 0.0975, 1.3991, 0.8791),
        (17, 1.0879, 0.3001, 1.5610, 0.8791),
        (18, 0.8338, 0.0566, 1.6791, 0.8774),
        (20, 0.4546, 0.0867, 2.0801, 0.8729),
        (21, 1.3623, 0.5118, 2.0075, 0.8734),
        (28, 1.3591, 0.2305, 3.5888, 0.8584),
        (29, 0.9718, 0.2563, 2.7060, 0.8676),
        (30, 0.4503, 0.0571, 0.9296, 0.8860),
        (31, 0.0523, 0.0090, 0.6270, 0.8883),
        (32, 0.0492, 0.0083, 0.5264, 0.8901),
        (33, 1.5472, 0.3850, 4.5078, 0.8495),
        (34, 0.5205, 0.1392, 2.1473, 0.8739),
        (35, 0.1538, 0.0555, 0.7814, 0.8824),
        (36, 0.0653, 0.0223, 0.5854, 0.8886),
        (37, 4.6949, 0.0488, 2.8603, 0.8667),
        (38, 2.9905, 0.1957, 5.1688, 0.8402),
        (39, 0.6683, 0.1811, 2.3110, 0.8719),
        (40, 0.1141, 0.0160, 0.3531, 0.8910),
        (41, 0.0132, 0.0056, 0.3390, 0.8917),
        (42, 0.0097, 0.0038, 0.3869, 0.8909)
    ]
    df_por = pd.DataFrame(t3_data, columns=[
        'Condition_ID', 'XCT_Porosity_Mean_pct', 'XCT_Porosity_Std_pct',
        'Archimedes_Porosity_Mean_pct', 'Archimedes_Porosity_Std_pct'
    ])

    # Table 4: Pore Quantity (35 conditions)
    t4_data = [
        (1, 44006, 44006, 0, 0, 0),
        (2, 57383, 57383, 0, 0, 0),
        (3, 57874, 57874, 0, 0, 0),
        (4, 67526, 67526, 0, 0, 0),
        (5, 93003, 93003, 0, 0, 0),
        (6, 62355, 62355, 0, 0, 0),
        (7, 56069, 56069, 0, 0, 0),
        (8, 50236, 50236, 0, 0, 0),
        (9, 35273, 35273, 0, 0, 0),
        (10, 22806, 22806, 0, 0, 0),
        (11, 17670, 17670, 0, 0, 0),
        (12, 42124, 42122, 2, 0, 0),
        (13, 41642, 41641, 1, 0, 0),
        (14, 31408, 31408, 0, 0, 0),
        (15, 48679, 48679, 0, 0, 0),
        (16, 29954, 29954, 0, 0, 0),
        (17, 40214, 40214, 0, 0, 0),
        (18, 44417, 44417, 0, 0, 0),
        (20, 28498, 28498, 0, 0, 0),
        (21, 38997, 38997, 0, 0, 0),
        (28, 68407, 68405, 1, 0, 1),
        (29, 53425, 53425, 0, 0, 0),
        (30, 22826, 22826, 0, 0, 0),
        (31, 2052, 2052, 0, 0, 0),
        (32, 1153, 1153, 0, 0, 0),
        (33, 76806, 76788, 6, 8, 4),
        (34, 40141, 40140, 0, 1, 0),
        (35, 16145, 16145, 0, 0, 0),
        (36, 7723, 7723, 0, 0, 0),
        (37, 11801, 10864, 691, 210, 36),
        (38, 3179, 2847, 180, 117, 35),
        (39, 46274, 46158, 91, 25, 0),
        (40, 11094, 11094, 0, 0, 0),
        (41, 2052, 2052, 0, 0, 0),
        (42, 288, 288, 0, 0, 0)
    ]
    df_pore_cnt = pd.DataFrame(t4_data, columns=[
        'Condition_ID', 'Pore_Count_All', 'Pore_Count_lt200um', 'Pore_Count_200_300um',
        'Pore_Count_300_500um', 'Pore_Count_gt500um'
    ])

    # Table 5: Defect Pore Size and Morphology (35 conditions)
    t5_data = [
        (1, 50, 15, 0.98, 0.04, 0.94, 3798, 0.76, 280),
        (2, 52, 16, 0.97, 0.05, 0.92, 3885, 0.60, 287),
        (3, 52, 17, 0.97, 0.05, 0.92, 4046, 1.55, 339),
        (4, 52, 18, 0.97, 0.05, 0.92, 4413, 1.14, 415),
        (5, 50, 17, 0.96, 0.06, 0.89, 4040, 1.72, 534),
        (6, 53, 19, 0.97, 0.05, 0.91, 4468, 1.17, 457),
        (7, 50, 15, 0.98, 0.05, 0.92, 3448, 0.48, 208),
        (8, 49, 14, 0.98, 0.05, 0.91, 3237, 0.52, 166),
        (9, 47, 13, 0.99, 0.04, 0.94, 3240, 0.40, 129),
        (10, 47, 12, 0.99, 0.03, 0.96, 2975, 0.78, 86),
        (11, 45, 11, 0.99, 0.03, 0.95, 3616, 1.29, 85),
        (12, 47, 12, 0.99, 0.03, 0.96, 3594, 0.33, 207),
        (13, 48, 13, 0.99, 0.03, 0.96, 3507, 0.67, 209),
        (14, 47, 13, 0.99, 0.03, 0.96, 3588, 0.52, 203),
        (15, 50, 15, 0.98, 0.04, 0.95, 3731, 0.63, 244),
        (16, 48, 13, 0.99, 0.03, 0.95, 3599, 0.99, 220),
        (17, 50, 15, 0.98, 0.04, 0.94, 3900, 0.85, 286),
        (18, 50, 16, 0.98, 0.04, 0.94, 4564, 0.94, 354),
        (20, 48, 15, 0.98, 0.04, 0.94, 4447, 1.18, 333),
        (21, 50, 16, 0.98, 0.04, 0.93, 4760, 1.49, 370),
        (28, 51, 16, 0.97, 0.05, 0.91, 4311, 0.80, 502),
        (29, 50, 16, 0.98, 0.05, 0.93, 3608, 0.62, 347),
        (30, 47, 13, 0.99, 0.03, 0.96, 4074, 0.69, 116),
        (31, 47, 17, 0.99, 0.03, 0.94, 6573, 0.20, 19),
        (32, 59, 27, 0.96, 0.05, 0.91, 6129, 0.20, 9),
        (33, 45, 13, 0.99, 0.04, 0.79, 4081, 0.33, 670),
        (34, 48, 14, 0.98, 0.04, 0.94, 3903, 0.43, 373),
        (35, 45, 11, 0.99, 0.02, 0.97, 3843, 0.70, 105),
        (36, 43, 10, 1.00, 0.02, 0.98, 4126, 0.20, 42),
        (37, 105, 69, 0.85, 0.12, 0.60, 21643, 1.96, 339),
        (38, 97, 101, 0.81, 0.20, 0.29, 7327, 5.55, 439),
        (39, 44, 17, 0.99, 0.04, 0.67, 3315, 0.24, 339),
        (40, 44, 11, 1.00, 0.02, 0.97, 3606, 0.20, 62),
        (41, 41, 9, 1.00, 0.01, 0.98, 13627, 0.20, 38),
        (42, 52, 16, 0.98, 0.04, 0.96, 30228, 0.35, 65)
    ]
    df_pore_morph = pd.DataFrame(t5_data, columns=[
        'Condition_ID', 'Pore_Eq_Diameter_Mean_um', 'Pore_Eq_Diameter_Std_um',
        'Pore_Sphericity_Mean', 'Pore_Sphericity_Std', 'Pore_Sphericity_Vol_Mean',
        'Projected_Area_RMS_Mean_um2', 'Projected_Area_RMS_Max_mm2', 'Projected_Area_RMS_Total_mm2'
    ])

    # Table 6: Surface Roughness (35 conditions)
    t6_data = [
        (1, 12.69, 1.80, 15.70, 2.20),
        (2, 11.99, 1.67, 14.36, 2.64),
        (3, 15.26, 3.16, 19.20, 3.19),
        (4, 14.38, 5.20, 18.71, 5.98),
        (5, 15.84, 4.15, 19.99, 4.31),
        (6, 16.07, 2.80, 20.15, 2.73),
        (7, 12.23, 2.45, 14.84, 1.76),
        (8, 11.49, 2.00, 14.44, 1.11),
        (9, 12.00, 2.73, 14.89, 2.55),
        (10, 10.83, 1.45, 14.02, 1.62),
        (11, 10.21, 1.26, 11.94, 1.06),
        (12, 10.66, 1.45, 12.60, 1.09),
        (13, 11.81, 0.31, 13.10, 0.94),
        (14, 10.19, 2.73, 12.77, 1.73),
        (15, 10.19, 2.73, 12.77, 1.73),
        (16, 11.07, 2.42, 13.57, 1.24),
        (17, 11.50, 2.63, 15.07, 2.11),
        (18, 14.47, 3.73, 17.46, 3.94),
        (20, 13.96, 1.98, 17.51, 1.52),
        (21, 15.53, 2.98, 18.75, 3.21),
        (28, 17.62, 7.66, 22.15, 8.66),
        (29, 13.78, 3.02, 16.80, 3.35),
        (30, 13.68, 2.11, 17.32, 0.97),
        (31, 13.62, 3.76, 16.54, 4.95),
        (32, 13.57, 4.29, 17.50, 5.13),
        (33, 12.31, 1.07, 14.71, 2.40),
        (34, 9.44, 2.35, 12.21, 0.80),
        (35, 10.67, 1.41, 12.98, 1.45),
        (36, 12.30, 1.40, 14.28, 2.11),
        (37, 11.13, 2.12, 13.32, 1.23),
        (38, 9.78, 2.85, 12.73, 3.23),
        (39, 9.74, 1.87, 12.09, 1.46),
        (40, 10.79, 1.01, 12.00, 0.82),
        (41, 11.51, 0.23, 12.34, 0.95),
        (42, 11.16, 2.07, 13.57, 1.93)
    ]
    df_rough = pd.DataFrame(t6_data, columns=[
        'Condition_ID', 'Roughness_Ra_Mean_um', 'Roughness_Ra_Std_um',
        'Roughness_RMS_Mean_um', 'Roughness_RMS_Std_um'
    ])

    # Table 7: Prior-beta Grain Morphology (42 conditions)
    t7_data = [
        (1, 204, 114, 1.92, 0.66),
        (2, 205, 105, 1.98, 0.74),
        (3, 188, 88, 1.88, 0.60),
        (4, 167, 63, 1.89, 0.58),
        (5, 168, 64, 1.89, 0.58),
        (6, 173, 63, 1.91, 0.61),
        (7, 171, 64, 1.89, 0.60),
        (8, 164, 62, 1.88, 0.60),
        (9, 185, 71, 1.88, 0.57),
        (10, 202, 105, 2.05, 0.74),
        (11, 251, 122, 1.95, 0.63),
        (12, 240, 128, 2.10, 0.79),
        (13, 222, 95, 1.99, 0.64),
        (14, 189, 67, 1.96, 0.68),
        (15, 208, 102, 2.06, 0.74),
        (16, 197, 95, 1.96, 0.69),
        (17, 229, 101, 1.93, 0.63),
        (18, 168, 55, 1.95, 0.62),
        (19, 179, 69, 1.96, 0.63),
        (20, 172, 67, 1.91, 0.61),
        (21, 224, 108, 1.88, 0.61),
        (22, 264, 115, 2.46, 1.02),
        (23, 177, 58, 1.93, 0.62),
        (24, 188, 74, 1.88, 0.58),
        (25, 211, 102, 1.89, 0.57),
        (26, 257, 129, 1.95, 0.60),
        (27, 196, 98, 1.88, 0.57),
        (28, 182, 75, 1.91, 0.59),
        (29, 191, 73, 1.89, 0.61),
        (30, 164, 58, 1.90, 0.59),
        (31, 207, 92, 1.91, 0.60),
        (32, 176, 77, 1.86, 0.57),
        (33, 210, 116, 1.92, 0.62),
        (34, 165, 63, 1.89, 0.57),
        (35, 223, 98, 1.93, 0.64),
        (36, 209, 91, 1.90, 0.58),
        (37, 224, 129, 1.89, 0.57),
        (38, 207, 96, 2.24, 0.94),
        (39, 231, 110, 1.97, 0.68),
        (40, 205, 101, 1.87, 0.57),
        (41, 228, 103, 1.93, 0.61),
        (42, 234, 113, 2.00, 0.67)
    ]
    df_grain = pd.DataFrame(t7_data, columns=[
        'Condition_ID', 'Grain_Eq_Diameter_Mean_um', 'Grain_Eq_Diameter_Std_um',
        'Grain_Aspect_Ratio_Mean', 'Grain_Aspect_Ratio_Std'
    ])

    # Table 8: Mechanical Properties (42 conditions)
    t8_data = [
        (1, 1021, 6, 942, 10, 6.15, 0.08, 10.70, 0.76),
        (2, 1008, 7, 931, 12, 5.81, 0.21, 9.19, 1.06),
        (3, 992, 16, 904, 35, 6.13, 0.64, 8.98, 0.69),
        (4, 1009, 20, 929, 21, 5.52, 0.17, 8.50, 1.33),
        (5, 995, 27, 914, 35, 5.37, 0.38, 7.23, 1.22),
        (6, 982, 14, 892, 44, 5.27, 0.32, 6.95, 0.54),
        (7, 1038, 8, 960, 13, 6.06, 0.25, 10.76, 0.78),
        (8, 1053, 9, 974, 19, 6.13, 0.16, 11.87, 1.03),
        (9, 1058, 7, 984, 13, 5.92, 0.26, 12.35, 0.91),
        (10, 1070, 10, 994, 14, 6.37, 0.47, 14.03, 0.62),
        (11, 1081, 17, 1000, 19, 6.33, 0.46, 13.91, 0.84),
        (12, 1091, 22, 1025, 18, 5.86, 0.42, 12.39, 2.05),
        (13, 1085, 21, 1014, 19, 6.45, 0.19, 12.53, 2.20),
        (14, 1076, 18, 1005, 15, 6.47, 0.22, 12.82, 0.96),
        (15, 1056, 7, 988, 16, 5.93, 0.67, 11.86, 1.41),
        (16, 1041, 15, 965, 18, 6.13, 0.28, 10.71, 1.33),
        (17, 1014, 15, 940, 25, 5.71, 0.17, 9.28, 0.57),
        (18, 1001, 7, 927, 17, 6.02, 0.32, 9.70, 0.85),
        (19, 1002, 12, 923, 15, 5.72, 0.48, 9.33, 0.81),
        (20, 998, 9, 921, 19, 5.77, 0.45, 9.32, 0.92),
        (21, 995, 9, 913, 10, 5.76, 0.29, 9.20, 0.70),
        (22, 969, 15, 894, 17, 5.43, 0.32, 7.57, 0.63),
        (23, 1015, 8, 936, 8, 5.69, 0.22, 8.03, 0.81),
        (24, 1038, 9, 955, 12, 5.99, 0.43, 12.46, 1.32),
        (25, 1054, 21, 977, 20, 5.82, 0.19, 13.58, 1.54),
        (26, 1037, 11, 962, 12, 5.84, 0.32, 7.88, 1.33),
        (27, 1015, 24, 946, 25, 4.21, 1.07, 4.47, 1.18),
        (28, 965, 8, 897, 7, 4.12, 1.39, 5.10, 2.19),
        (29, 1003, 12, 926, 15, 5.53, 0.23, 7.79, 0.51),
        (30, 1036, 8, 955, 13, 6.24, 0.31, 12.39, 1.09),
        (31, 1055, 12, 975, 13, 5.97, 0.28, 12.50, 3.60),
        (32, 1033, 15, 959, 18, 6.00, 0.36, 10.81, 2.73),
        (33, 954, 30, 901, 20, 3.30, 2.36, 4.38, 3.78),
        (34, 1046, 12, 972, 8, 5.69, 0.35, 8.55, 1.49),
        (35, 1074, 19, 993, 22, 6.07, 0.28, 13.46, 1.52),
        (36, 1075, 13, 997, 12, 6.08, 0.22, 14.09, 1.88),
        (37, 960, 73, 902, 80, 2.76, 0.87, 3.01, 1.09),
        (38, 849, 119, 829, 113, 1.23, 0.23, 1.26, 0.29),
        (39, 1040, 33, 989, 33, 2.71, 0.56, 3.05, 0.78),
        (40, 1111, 26, 1047, 23, 6.29, 0.42, 14.22, 1.40),
        (41, 1113, 25, 1048, 24, 6.23, 0.41, 15.38, 1.30),
        (42, 1079, 37, 1014, 29, 5.56, 1.63, 11.77, 5.47)
    ]
    df_mech = pd.DataFrame(t8_data, columns=[
        'Condition_ID', 'Yield_Strength_Mean_MPa', 'Yield_Strength_Std_MPa',
        'UTS_Mean_MPa', 'UTS_Std_MPa',
        'Elongation_Necking_Mean_pct', 'Elongation_Necking_Std_pct',
        'Elongation_Fracture_Mean_pct', 'Elongation_Fracture_Std_pct'
    ])

    # Table 9: Hardness and Elastic Modulus (42 conditions)
    t9_data = [
        (1, 358, 26, 112, 5),
        (2, 375, 14, 108, 4),
        (3, 354, 10, 109, 12),
        (4, 378, 19, 108, 4),
        (5, 371, 10, 113, 11),
        (6, 378, 12, 128, 43),
        (7, 388, 29, 113, 7),
        (8, 372, 15, 117, 10),
        (9, 380, 26, 113, 3),
        (10, 350, 11, 119, 6),
        (11, 384, 30, 118, 6),
        (12, 373, 19, 116, 7),
        (13, 369, 6, 114, 6),
        (14, 378, 20, 111, 9),
        (15, 361, 20, 116, 7),
        (16, 372, 16, 114, 5),
        (17, 390, 27, 109, 4),
        (18, 391, 25, 108, 7),
        (19, 385, 19, 106, 6),
        (20, 384, 16, 107, 10),
        (21, 388, 23, 104, 6),
        (22, 382, 10, 102, 2),
        (23, 359, 25, 109, 6),
        (24, 374, 12, 112, 4),
        (25, 390, 22, 113, 5),
        (26, 384, 21, 109, 5),
        (27, 365, 14, 108, 5),
        (28, 370, 10, 104, 8),
        (29, 388, 19, 108, 6),
        (30, 387, 21, 110, 5),
        (31, 377, 19, 113, 7),
        (32, 380, 33, 109, 3),
        (33, 375, 17, 106, 8),
        (34, 388, 28, 111, 3),
        (35, 389, 25, 118, 5),
        (36, 386, 23, 114, 6),
        (37, 420, 15, 104, 11),
        (38, 385, 20, 101, 11),
        (39, 372, 19, 116, 6),
        (40, 378, 17, 118, 8),
        (41, 386, 28, 117, 6),
        (42, 400, 10, 116, 8)
    ]
    df_hard = pd.DataFrame(t9_data, columns=[
        'Condition_ID', 'Vickers_Hardness_Mean_HV', 'Vickers_Hardness_Std_HV',
        'Youngs_Modulus_Mean_GPa', 'Youngs_Modulus_Std_GPa'
    ])

    # Merge into Master Dataset (all 42 conditions)
    df_master = df_proc.merge(df_grain, on='Condition_ID', how='left')
    df_master = df_master.merge(df_mech, on='Condition_ID', how='left')
    df_master = df_master.merge(df_hard, on='Condition_ID', how='left')
    df_master = df_master.merge(df_por, on='Condition_ID', how='left')
    df_master = df_master.merge(df_pore_cnt, on='Condition_ID', how='left')
    df_master = df_master.merge(df_pore_morph, on='Condition_ID', how='left')
    df_master = df_master.merge(df_rough, on='Condition_ID', how='left')

    # Flag for XCT availability (35 conditions have XCT, 7 conditions 19, 22-27 do not)
    df_master['Has_XCT_Data'] = df_master['XCT_Porosity_Mean_pct'].notna()

    # Process defect regimes classification based on literature and energy density:
    # LOF (Lack of Fusion): LED < 250 J/m or VED < 50 J/mm^3
    # Keyholing: LED > 400 J/m and high power or Pv > 300
    # Conduction / Optimal: Balanced energy density
    def assign_regime(row):
        led = row['LED_J_m']
        por = row['XCT_Porosity_Mean_pct'] if pd.notna(row['XCT_Porosity_Mean_pct']) else row['Archimedes_Porosity_Mean_pct']
        if led < 260 or row['Laser_Power_W'] <= 115:
            return 'Lack of Fusion'
        elif led >= 415 or row['Pv_W_m_s'] >= 600:
            return 'Keyhole / Balling'
        else:
            return 'Optimal / Conduction'

    df_master['Process_Defect_Regime'] = df_master.apply(assign_regime, axis=1)

    return df_master

def process_raw_pore_files():
    files = glob.glob('dataset/*.csv')
    print(f'Processing {len(files)} raw micro-CT CSV files...')
    summary_records = []

    for f in sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())):
        cid = int(re.search(r'\d+', os.path.basename(f)).group())
        try:
            # Read header and skip row 0 if it contains subheaders
            df_sample = pd.read_csv(f, skiprows=1, header=None)
            # Typically 5 columns: index, volume, eq_diameter, surface_area, sphericity
            if df_sample.shape[1] >= 5:
                df_sample.columns = ['pore_idx', 'volume_mm3', 'eq_dia_mm', 'surf_area_mm2', 'sphericity'][:df_sample.shape[1]]
            else:
                continue

            for col in ['volume_mm3', 'eq_dia_mm', 'surf_area_mm2', 'sphericity']:
                df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')

            df_sample = df_sample.dropna(subset=['volume_mm3', 'sphericity'])
            n_pores = len(df_sample)
            vol_sum = df_sample['volume_mm3'].sum()
            vol_mean = df_sample['volume_mm3'].mean()
            vol_median = df_sample['volume_mm3'].median()
            dia_mean_um = (df_sample['eq_dia_mm'] * 1000).mean()
            dia_p95_um = (df_sample['eq_dia_mm'] * 1000).quantile(0.95)
            sph_mean = df_sample['sphericity'].mean()
            sph_median = df_sample['sphericity'].median()

            summary_records.append({
                'Condition_ID': cid,
                'Raw_Pore_Count': n_pores,
                'Total_Pore_Volume_mm3': round(vol_sum, 6),
                'Mean_Pore_Volume_mm3': round(vol_mean, 8),
                'Median_Pore_Volume_mm3': round(vol_median, 8),
                'Mean_Pore_Diameter_um': round(dia_mean_um, 2),
                'P95_Pore_Diameter_um': round(dia_p95_um, 2),
                'Mean_Pore_Sphericity': round(sph_mean, 4),
                'Median_Pore_Sphericity': round(sph_median, 4)
            })
        except Exception as e:
            print(f'Error reading {f}: {e}')

    df_raw_sum = pd.DataFrame(summary_records)
    return df_raw_sum

def build_data_dictionary():
    dict_data = [
        # Process
        ('Condition_ID', 'Identifier', 'Integer', '1-42', 'Process parameter combination index', 'Prior fabrication designed index'),
        ('Laser_Power_W', 'Process (Group 1)', 'W', '75 - 500', 'Laser power applied during LPBF internal scanning', 'ProX DMP 320 parameter setting'),
        ('Scan_Speed_mm_s', 'Process (Group 1)', 'mm/s', '400 - 1600', 'Galvanometer laser scanning speed', 'ProX DMP 320 parameter setting'),
        ('LED_J_m', 'Process (Group 1)', 'J/m', '188 - 469', 'Linear energy density (P / v)', 'Analytical physics descriptor'),
        ('MED_J_sqrt_m_s', 'Process (Group 1)', 'J/sqrt(m)/s', '119 - 440', 'Modified volumetric energy density C * P / sqrt(v)', 'Analytical thermal penetration descriptor'),
        ('Pv_W_m_s', 'Process (Group 1)', 'W*m/s', '30 - 760', 'Laser power multiplied by scan speed (melt pool elongation / balling factor)', 'Analytical melt pool stability descriptor'),
        ('VED_J_mm3', 'Process (Group 1)', 'J/mm^3', '38.2 - 146.3', 'Volumetric energy density P / (v * h * t) with h=82um, t=60um', 'Analytical energy density descriptor'),
        ('Process_Defect_Regime', 'Process Category', 'Categorical', 'Lack of Fusion / Optimal / Keyhole', 'Manufacturing defect regime classification', 'Metallurgical process map regime'),
        
        # Structure - Pores
        ('XCT_Porosity_Mean_pct', 'Structure/Defect (Group 2, Target 1)', '%', '0.0097 - 4.6949', 'Internal pore volume fraction measured by X-ray computed tomography', 'GE Phoenix v|tome|x L300 micro-CT (10 um voxel)'),
        ('XCT_Porosity_Std_pct', 'Structure/Defect Uncertainty', '%', '0.0003 - 0.5489', 'Standard deviation of XCT porosity measurement', 'Micro-CT image analysis variance'),
        ('Archimedes_Porosity_Mean_pct', 'Structure/Defect (Group 2)', '%', '0.3390 - 5.1688', 'Apparent porosity measured by Archimedes immersion method', 'OHAUS EX623 analytical balance (5 tests/specimen)'),
        ('Archimedes_Porosity_Std_pct', 'Structure/Defect Uncertainty', '%', '0.8402 - 0.8917', 'Standard deviation of Archimedes porosity', 'Immersion measurement variance'),
        ('Pore_Count_All', 'Structure/Defect (Group 2)', 'Count', '288 - 93003', 'Total number of detected pores in XCT gauge section', 'Avizo 3D pore segmentation'),
        ('Pore_Count_lt200um', 'Structure/Defect (Group 2)', 'Count', '288 - 93003', 'Number of pores with equivalent diameter < 200 um', 'Avizo size-binned segmentation'),
        ('Pore_Count_200_300um', 'Structure/Defect (Group 2)', 'Count', '0 - 691', 'Number of pores with equivalent diameter 200-300 um (critical defects)', 'Avizo size-binned segmentation'),
        ('Pore_Count_300_500um', 'Structure/Defect (Group 2)', 'Count', '0 - 210', 'Number of pores with equivalent diameter 300-500 um (severe defects)', 'Avizo size-binned segmentation'),
        ('Pore_Count_gt500um', 'Structure/Defect (Group 2)', 'Count', '0 - 36', 'Number of pores with equivalent diameter > 500 um (macroscopic lack-of-fusion)', 'Avizo size-binned segmentation'),
        ('Pore_Eq_Diameter_Mean_um', 'Structure/Defect (Group 2)', 'um', '41 - 105', 'Mean equivalent spherical diameter of internal pores', 'Avizo 3D morphometry'),
        ('Pore_Sphericity_Mean', 'Structure/Defect (Group 2)', 'Dimensionless', '0.81 - 1.00', 'Mean numerical pore sphericity (Psi = pi^(1/3)*(6V)^(2/3)/A)', 'Avizo shape morphometry'),
        ('Pore_Sphericity_Vol_Mean', 'Structure/Defect (Group 2)', 'Dimensionless', '0.29 - 0.98', 'Volume-weighted pore sphericity (highlights irregular large LOF pores)', 'Avizo weighted morphometry'),
        ('Projected_Area_RMS_Mean_um2', 'Structure/Defect (Group 2)', 'um^2', '2975 - 30228', 'Mean RMS projected pore area orthogonal to tensile loading axis', 'Image analysis on load plane cross-section'),
        ('Projected_Area_RMS_Max_mm2', 'Structure/Defect (Group 2)', 'mm^2', '0.20 - 5.55', 'Maximum projected pore area on cross-section plane', 'Image analysis on load plane cross-section'),
        ('Projected_Area_RMS_Total_mm2', 'Structure/Defect (Group 2)', 'mm^2', '9 - 670', 'Total projected pore area on cross-section plane', 'Image analysis on load plane cross-section'),
        
        # Structure - Surface & Grain
        ('Roughness_Ra_Mean_um', 'Structure/Surface (Group 3)', 'um', '9.44 - 17.62', 'Arithmetic mean surface roughness Ra on as-built surfaces', 'Keyence optical profiler'),
        ('Roughness_RMS_Mean_um', 'Structure/Surface (Group 3)', 'um', '11.94 - 22.15', 'Root mean square surface roughness RMS', 'Keyence optical profiler'),
        ('Grain_Eq_Diameter_Mean_um', 'Structure/Microstructure (Group 3)', 'um', '164 - 264', 'Mean prior-beta grain equivalent diameter', 'Optical microscopy + MIPAR segmentation'),
        ('Grain_Aspect_Ratio_Mean', 'Structure/Microstructure (Group 3)', 'Dimensionless', '1.86 - 2.46', 'Mean columnar aspect ratio of prior-beta grains along build direction', 'Optical microscopy + MIPAR segmentation'),
        
        # Properties (Targets)
        ('Yield_Strength_Mean_MPa', 'Mechanical Property (Target 2)', 'MPa', '849 - 1113', '0.2% offset tensile yield strength (sigma_y)', 'MTS Criterion 45 + DIC (ASTM E8/E8M)'),
        ('Yield_Strength_Std_MPa', 'Property Uncertainty', 'MPa', '6 - 119', 'Standard deviation of yield strength across 4 tensile specimens', 'Tensile test variance (2 flat + 2 round)'),
        ('UTS_Mean_MPa', 'Mechanical Property (Target 3)', 'MPa', '829 - 1048', 'Ultimate tensile strength (sigma_UTS)', 'MTS Criterion 45 + DIC (ASTM E8/E8M)'),
        ('UTS_Std_MPa', 'Property Uncertainty', 'MPa', '7 - 113', 'Standard deviation of UTS across 4 tensile specimens', 'Tensile test variance (2 flat + 2 round)'),
        ('Elongation_Fracture_Mean_pct', 'Mechanical Property (Target 4)', '%', '1.26 - 15.38', 'Tensile elongation to fracture / total ductility (epsilon_f)', 'MTS extensometer (24 mm gauge) + DIC'),
        ('Elongation_Fracture_Std_pct', 'Property Uncertainty', '%', '0.29 - 5.47', 'Standard deviation of elongation to fracture across 4 specimens', 'Tensile test variance'),
        ('Elongation_Necking_Mean_pct', 'Mechanical Property', '%', '1.23 - 6.47', 'Uniform elongation to onset of necking (epsilon_u)', 'MTS extensometer + DIC'),
        ('Vickers_Hardness_Mean_HV', 'Mechanical Property (Target 5)', 'HV', '350 - 420', 'Vickers microhardness (100 gf load, 10 s dwell, 10 indents/condition)', 'LECO LM 110AT microhardness tester'),
        ('Youngs_Modulus_Mean_GPa', 'Mechanical Property', 'GPa', '101 - 128', 'Tensile elastic modulus E', 'MTS Criterion 45 + DIC'),
        ('Has_XCT_Data', 'Data Flag', 'Boolean', 'True (35) / False (7)', 'Indicator whether XCT pore scan data exists for condition', 'Experimental campaign protocol')
    ]
    df_dict = pd.DataFrame(dict_data, columns=[
        'Feature_Name', 'Role_and_Group', 'Physical_Units', 'Observed_Range', 'Description', 'Experimental_Source'
    ])
    return df_dict

if __name__ == '__main__':
    print('Building master dataset...')
    df_master = build_tables()
    df_raw = process_raw_pore_files()
    df_dict = build_data_dictionary()

    df_master.to_csv('lpbf_ti6al4v_psp_dataset.csv', index=False)
    df_raw.to_csv('lpbf_raw_pore_aggregated.csv', index=False)
    df_dict.to_csv('lpbf_ti6al4v_data_dictionary.csv', index=False)

    print('Successfully created:')
    print(f'1. lpbf_ti6al4v_psp_dataset.csv (Shape: {df_master.shape})')
    print(f'2. lpbf_raw_pore_aggregated.csv (Shape: {df_raw.shape})')
    print(f'3. lpbf_ti6al4v_data_dictionary.csv (Shape: {df_dict.shape})')
    print('\nMissing value audit:')
    print(df_master.isna().sum()[df_master.isna().sum() > 0])
