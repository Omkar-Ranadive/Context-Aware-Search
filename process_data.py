import pandas as pd


def load_files(file_list):

    # Store contents in a list
    articles = [pd.read_csv(f) for f in file_list]

    # Separate out titles and content
    # This step will depend on the dataset used, so will vary from dataset to dataset

    # First, take out the titles from every file
    titles = [article['title'] for article in articles]

    # Then the actual content
    contents = [article['content'] for article in articles]

    return titles, contents


def load_headers(file_list):

    # Store contents in a list
    articles = [pd.read_csv(f) for f in file_list]

    # Load only the headers
    headers = [article['title'] for article in articles]

    return headers



if __name__ == '__main__':
    titles, contents = load_files(['articles1.csv'])

    print(contents[0][0])