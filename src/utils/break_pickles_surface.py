import pickle
import os


def save_data_as_pickle(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, value in data.items():
        if key in ["x_list", "x_dot_list"]:
            with open(
                os.path.join(output_dir, f"df_surface_up_sindy_{key}.pkl"), "wb"
            ) as pickle_file:
                pickle.dump(value, pickle_file)


file_path = "./df_surface_up_sindy.pkl"
output_directory = "./output_data"

with open(file_path, "rb") as file:
    loaded_dictionary = pickle.load(file)

# Save each key's content as a separate pickle file
save_data_as_pickle(loaded_dictionary, output_directory)

print(f"Data saved to {output_directory}")
