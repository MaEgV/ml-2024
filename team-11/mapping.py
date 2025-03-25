from sklearn.preprocessing import OneHotEncoder


# one hot encoding mapping
building_type_encoder = OneHotEncoder(
    categories=[[0, 1, 2, 3, 4, 5, 6]],
    sparse_output=False,
    drop='first'
)

building_type_column_names = [
        # 'building_type_Dont_know',
        'building_type_Other',
        'building_type_Panel',
        'building_type_Monolithic',
        'building_type_Brick',
        'building_type_Blocky',
        'building_type_Wooden'
    ]

object_type_encoder = OneHotEncoder(
        categories=[[0, 2]],
        sparse_output=False,
        drop='first'
    )

object_type_column_names = [
    # 'object_type_Secondary',
    'object_type_New_Building'
]


# only for ui app select-box

object_type_str_to_int = {
    "Вторичка": 0,
    "Новостройка": 2
}

object_type_int_to_str = {
    0: "Вторичка",
    2: "Новостройка"
}

building_type_str_to_int = {
    "Любой": 0,
    "Другой": 1,
    "Панельный": 2,
    "Монолитный": 3,
    "Кирпичный": 4,
    "Блочный": 5,
    "Деревянный": 6
}

building_type_int_to_str = {
    0: "Любой",
    1: "Другой",
    2: "Панельный",
    3: "Монолитный",
    4: "Кирпичный",
    5: "Блочный",
    6: "Деревянный"
}