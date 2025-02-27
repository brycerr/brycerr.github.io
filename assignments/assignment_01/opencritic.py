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
        return soup.find_all('div', class_='game-row'), response.text
    else:
        print(f"[{time.ctime()}] Failed to retrieve {url}: {response.status_code}")
        return [], None


def extract_game_info(game):
    # print(game)

    title_tag = game.find('div', class_='game-name').find('a')
    if title_tag:
        title = title_tag.text.strip()
    else:
        title = 'N/A'
    # print(title)

    platform_tag = game.find('div', class_='platforms')
    if platform_tag:
        platform = platform_tag.text.strip()
    else:
        platform = 'N/A'

    release_tag = game.find('div', class_='first-release-date')
    if release_tag:
        if release_tag.find_all('span'):
            release_date = release_tag.find_all('span')[0].text.strip()
        else:
            release_date = 'N/A'
    else:
        release_date = 'N/A'

    score_tag = game.find('div', class_='inner-orb')
    if score_tag:
        score = score_tag.text.strip()
    else:
        score = 'N/A'

    return [title, platform, release_date, score]


def main():
    base_url = 'https://opencritic.com/browse/all?page='
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
        save_html(game_html[1], 'html-data/opencritic', f'page_{page}.html')

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

    # record every game in table B
    print(f"[{time.ctime()}] Writing to table B...")

    with open('tableB.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Title', 'Platform', 'Release_Date', 'Score'])
        for i, game in enumerate(games):
            writer.writerow([i+1, game[0], game[1], game[2], game[3]])

    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"[{time.ctime()}] Extracted {len(games)} games to table B in {time_elapsed} seconds.")


if __name__ == '__main__':
    main()
