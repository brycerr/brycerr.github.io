import os
import requests
from bs4 import BeautifulSoup
import csv
import time


def save_html(content, directory, filename):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
        file.write(content)


def fetch_games(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.find_all('div', class_='c-finderProductCard'), response.text
    else:
        print(f"[{time.ctime()}] Failed to retrieve {url}: {response.status_code}")
        return [], None


def extract_game_info(game):
    title_tag = game.find('div', class_='c-finderProductCard_title')
    if title_tag:
        title = title_tag.get('data-title', 'N/A')
    else:
        title = 'N/A'
    # print(title)

    # platform_tag = game.find('div', class_='c-finderProductCard_title')
    # if platform_tag:
    #     platform = platform_tag.get('data-title')
    # else:
    #     platform = 'N/A'
    platform = 'N/A'

    meta_tag = game.find('div', class_='c-finderProductCard_meta')
    if meta_tag:
        if meta_tag.find_all('span'):
            release_date = meta_tag.find_all('span')[0].text.strip()
        else:
            release_date = 'N/A'

        metascore_tag = game.find('div', class_='c-siteReviewScore')
        if metascore_tag and metascore_tag.find('span'):
            metascore = metascore_tag.find('span').text.strip()
        else:
            metascore = 'N/A'
    else:
        release_date = metascore = 'N/A'

    return [title, platform, release_date, metascore]


def main():
    base_url = 'https://www.metacritic.com/browse/game/?page='
    headers = {'User-Agent': 'Mozilla/5.0'}

    games = []          # list of all extracted games
    num_games = 2500    # total number of games to extract
    page = 1

    # record elapsed time
    time_start = time.time()

    print(f"[{time.ctime()}] Starting to extract games...")

    while len(games) < num_games:
        url = f'{base_url}{page}'
        game_html = fetch_games(url, headers)
        game_cards = game_html[0]

        # save each html webpage
        save_html(game_html[1], 'html-data/metacritic', f'page_{page}.html')

        if not game_cards:
            print(f"[{time.ctime()}] Exiting on page {page}, no more games found.")
            break  # Exit if no more games are found

        for game in game_cards:
            games.append(extract_game_info(game))
            if len(games) >= num_games:
                break

            if len(games) % 100 == 0:
                print(f"[{time.ctime()}] {len(games)} games extracted.")
        page += 1

    # record every game in table A
    print(f"[{time.ctime()}] Writing to table A...")

    with open('tableA.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Title', 'Platform', 'Release_Date', 'Metascore'])
        for i, game in enumerate(games):
            writer.writerow([i+1, game[0], game[1], game[2], game[3]])

    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"[{time.ctime()}] Extracted {len(games)} games to table A in {time_elapsed} seconds.")


if __name__ == '__main__':
    main()
