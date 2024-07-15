import ast

# Test_string_dict
dict_string = "{'name': 'John', 'age': 30, 'city': 'New York'}"

# Convert string to dict
dictionary = ast.literal_eval(dict_string)

print(dictionary)
