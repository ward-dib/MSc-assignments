# Bootcamp exercises Q3:
''' Make a dictionary that contains a set of key:value pairs describing
some data of your choice. '''

personal_data = {'age':21, 'weight':12, 'height':171}

# To get a list of content: 
    
values = personal_data.keys()
print(values)

# To add to the dictionary:

personal_data ['eye_colour'] = 'brown'

# To access specific items:

print(personal_data ['eye_colour'])

# Other examples:
    
# 1- Dictionary of currently airing shows I'm keeping up with.

currently_airing = {'action': "jujutsu kaisen", 'sports': "haikyu", 'fantasy':
                   "burn the witch"} 
    
print(currently_airing ['action'])

# 2- Dictionary of the weight of crops harvested this year in tonnes.

fruits_2020 = {'apples': 476000, 'bananas': 115000000, 
              'oranges': 265000, 'kiwis': 120000}

fruit_import = fruits_2020.keys()
print(fruit_import)

fruits_2020 ['mangoes'] = 56205
