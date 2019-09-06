import argparse
from bs4 import BeautifulSoup
import io
import json
import nbformat
import os
import re, urlparse
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sqlite3
import subprocess
import sys
import time
import traceback
import wget


class Scraper(object):
    """ Generic scraper for project """

    def __init__(self):
        self.browser = None

    def get_browser(self):
        if self.browser is None:
            self.browser = webdriver.Firefox()
        return self.browser

    def parse_source(self, page_source):
        return BeautifulSoup(page_source, 'html5lib')


class KaggleCrawler(Scraper):
    """ Crawls kaggle.com/kernels to download new notebooks """

    def __init__(self, language, sort_by):
        """
    :param language: Language for notebooks [Python, R, Julia, SQLlite]
    :param sort_by: Sorting criteria [Most votes, Hotness, Most recent]
    """
        super(KaggleCrawler, self).__init__()
        self.downloaded = set()
        self.failed = set()
        self.language = language
        self.sort_by = sort_by

    def __repr__(self):
        return "KaggleCrawler(language=%s, sorted_by=%s)" % (
            self.language, self.sort_by
        )

    KAGGLE_BASE = 'https://www.kaggle.com/'

    DROP_DOWN_MENUS = {
        'language':
        set(['All Languages', 'Julia', 'R', 'Python', 'SQLite']),
        'sort':
        set(['Hotness', 'Most Recent', 'Most Votes']),
        'output':
        set(['All Output Types', 'Notebook', 'Visualization', 'Data', 'Other'])
    }

    def _get_dropdown_options(self, which):
        """ Possible options in particular drop down menu """
        if not which in KaggleCrawler.DROP_DOWN_MENUS.keys():
            raise ValueError(
                "Drop down menu must be one of %s: %s" %
                (KaggleCrawler.DROP_DOWN_MENUS.keys(), which)
            )
        return KaggleCrawler.DROP_DOWN_MENUS[which]

    def _get_drop_down(self, which):
        """ Find dropdown in website by identifying one of the options """
        # retrieve all possible drop downs
        dropdowns = self.browser.find_elements_by_xpath(
            "//div[contains(@class, 'Select-value')]"
        )
        options = self._get_dropdown_options(which)
        for d in dropdowns:
            if d.text in options:
                return d
        raise Exception("No matching options in drop downs")

    def _select_drop_down(self, which, option):
        """ Scroll a given drop down to a particular option. True if scroll succesfull false otherwise """
        dropdown = self._get_drop_down(which)
        options = self._get_dropdown_options(which)
        if not option.upper() in map(lambda x: x.upper(), options):
            raise ValueError(
                "No such option for menu %s: %s" % (which, option)
            )
        while True:
            # ignore case
            if dropdown.text.upper() == option.upper():
                return True
            else:
                # try next option
                dropdown.click()
                dropdown.send_keys(Keys.ARROW_DOWN)
                dropdown.send_keys(Keys.ENTER)
        return False

    def _sort_by(self, criteria):
        """ Sort kernels by given criteria """
        assert (self._select_drop_down('sort', criteria))

    def _select_language(self, lang):
        """ Filter kernels to a given language """
        assert (self._select_drop_down('language', lang))

    def _select_output(self, output):
        """ Filter kernels to particular output, currently just supports notebooks """
        assert (output.upper() == 'NOTEBOOK')
        assert (self._select_drop_down('output', output))

    def _get_height(self):
        """ Get height of scroll """
        return self.browser.execute_script("return document.body.scrollHeight")

    def _scroll_pg_down(self):
        """ Raw scroll, returns after executing, even if loading is not done """
        # scroll by entire page
        self.browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        # and scroll to see loading message
        self.browser.execute_script(
            "window.scrollTo(0, document.body.scrollHeight - 10);"
        )

    def _blocked_scroll_down(self):
        """ Blocked scroll down. Checks that message for loading more kernels appears and disappears """
        try:
            # wait until kernel loading message appears/disappears, timeout at 10s
            delay = 5
            wait = WebDriverWait(self.browser, delay)
            kernels_loading_msg = (
                By.XPATH, "//*[. = 'Loading more kernels...']"
            )
            # raw full page scroll
            self._scroll_pg_down()
            # wait until visible
            wait.until(EC.visibility_of_element_located(kernels_loading_msg))
            wait.until_not(
                EC.visibility_of_element_located(kernels_loading_msg)
            )
        except TimeoutException:
            print "Timedout on scroll!"
            pass

    def _infinite_scroll_kernels(self, n_scrolls=None, batch_size=10):
        """
    Scroll infinite scroll of kernels down by n_scrolls if provided or until height doesn't change.
    Prints count of scrolls every batch_size. Returns True if scrolled down
    """
        if n_scrolls is not None and n_scrolls <= 0:
            raise ValueError("Must scroll at least once: %d" % n_scrolls)
        curr = 0
        while n_scrolls is None or curr < n_scrolls:
            if curr % batch_size == 0:
                print "Scroll: %d" % curr
            current_height = self._get_height()
            self._blocked_scroll_down()
            new_height = self._get_height()
            if current_height == new_height:
                print "Height unchanged"
                return False
            curr += 1
        return True

    def _get_notebook_links(self):
        """
    Retrieve full notebook links on current kernels site.
    Only returns new links (ignores previously downloaded).
    """
        parsed_src = self.parse_source(self.browser.page_source)
        raw_links = parsed_src.findAll('a', {'class': 'scripts__item-name'})
        raw_links = set([
            urlparse.urljoin(KaggleCrawler.KAGGLE_BASE, link['href'])
            for link in raw_links
        ])
        # remove links we've already downloaded
        new_links = raw_links - self.downloaded - self.failed
        return list(new_links)

    def populate_downloaded(self, db_conn, table, col='link'):
        """
    Populate link of notebooks previously downloaded from a database.
    :param db_conn: Connection to existing database
    :param table: Name of table to query
    :param col: Optional column name to query, defaults to link
    :returns: Count of links from database
    """
        db_cursor = db_conn.cursor()
        db_cursor.execute("select distinct %s from %s" % (col, table))
        downloads = db_cursor.fetchall()
        db_cursor.close()
        for d in downloads:
            self.downloaded.add(d[0])
        return len(downloads)

    def _start_kernels(self):
        browser = self.get_browser()
        # size is important: otherwise we can end up with the mobile version
        browser.set_window_size(1000, 500)
        kernels_site = urlparse.urljoin(KaggleCrawler.KAGGLE_BASE, 'kernels/')
        browser.get(kernels_site)

    def _insert_into_db(self, db_conn, table, kernel):
        db_cursor = db_conn.cursor()
        try:
            nvalues = len(kernel.values())
            insert_cmd = 'INSERT INTO %s VALUES(%s)' % (
                table, ','.join(['?'] * nvalues)
            )
            db_cursor.execute(insert_cmd, kernel.values())
            # commit after each valid insert and only then add to downloaded
            db_conn.commit()
            self.downloaded.add(kernel.link)
        except sqlite3.ProgrammingError as err:
            # we can try to fix these for notebooks when inserting into sqlite, all else, SOL
            if kernel.file_ext == 'ipynb':
                kernel._convert_notebook()
                # and try again (with new file extension and code)
                self._insert_into_db(db_conn, table, kernel)
            else:
                # if not ipython, not chance at fixing, so just err out
                raise
        except Exception as err:
            raise
        finally:
            db_cursor.close()

    def crawl(self, db_conn, table, n=None):
        """
    Crawl kaggle.com/kernels and download notebooks.
    :param db_conn: Connection to existing database holding notebooks.
    :param table: Name of table to insert new notebooks into.
    :param n: Number of notebooks to download. Defaults to None which just means it
      never stops until terminated by user.
    """

        # some defaults
        output = 'notebook'
        # infinite scrolling params
        n_scrolls = 10
        batch_size = 5

        # counts number of notebooks remaining to be downloaded
        need = n
        # browse to initial kernels site
        self._start_kernels()
        # notebook filters
        self._select_language(self.language)
        self._select_output(output)
        self._sort_by(self.sort_by)
        # have separate browser to download individual kernels
        # we want to avoid losing infinite scroll on main crawler browser
        kernel_browser = Scraper().get_browser()
        # we can run out of kernels to scroll down to, so just stop after failing a lot
        scroll_failure_budget = 10

        while (n is None or need > 0) and scroll_failure_budget > 0:
            # scroll to find new links
            scrolled = self._infinite_scroll_kernels(
                n_scrolls=n_scrolls, batch_size=batch_size
            )
            scroll_failure_budget -= 1 if not scrolled else 0
            new_links = self._get_notebook_links()
            if len(new_links) > need:
                new_links = new_links[:need]
            for link in new_links:
                k = KaggleKernel(link)
                # use same browser, to avoid opening a ton of firefox windows
                try:
                    k.get(browser=kernel_browser)
                    self._insert_into_db(db_conn, table, k)
                    need -= 1
                except Exception as err:
                    print "Failed to download %s" % link
                    traceback.print_exc()
                    # keep track of failures to avoid repeating
                    self.failed.add(link)
                    pass
            print "%d/%d done" % (n - need, n)


class KaggleKernel(Scraper):
    def __init__(self, link):
        super(KaggleKernel, self).__init__()
        self.link = link
        self.code = None
        self.user = None
        self.votes = None
        self.language = None
        self.file_ext = None

    def __repr__(self):
        return "KaggleKernel(%s)" % self.link

    def get(self, browser=None):
        code_link = urlparse.urljoin(self.link + '/', 'code')
        print "Downloading: %s" % code_link
        if browser is None:
            browser = self.get_browser()
        browser.get(code_link)
        parsed_src = self.parse_source(browser.page_source)
        code = self.parse_code(parsed_src)
        user = self.parse_user(parsed_src)
        votes = self.parse_votes(parsed_src)
        return {'code': code, 'user': user, 'votes': votes}

    def parse_code(self, src):
        # now extract code text directly (rather than download as separate file)
        r = re.compile('script-code-pane__download.*')
        code_download_link = src.find('a', {'class': r})['href']
        ext, contents = self._parse_raw_code(code_download_link)
        # keep track of file extension, can be useful later on
        self.file_ext = ext
        try:
            if ext == 'ipynb':
                self.code = self._parse_notebook_code(contents)
                self.language = 'IPython Notebook'
            elif ext == 'py':
                self.code = self._parse_script_code(contents)
                self.language = 'Python'
            else:
                raise Exception(
                    "Need to implement parsing for extension: %s" % ext
                )
        except Exception as e:
            print e.message
        return self.code

    def _parse_notebook_code(self, txt):
        # strip output cells from notebook
        return self._strip_notebook_output(txt)

    # from https://gist.github.com/minrk/6176788
    def _cells(self, nb):
        """Yield all cells in an nbformat-insensitive manner"""
        if nb.nbformat < 4:
            for ws in nb.worksheets:
                for cell in ws.cells:
                    yield cell
        else:
            for cell in nb.cells:
                yield cell

    def _strip_output(self, nb):
        """strip the outputs from a notebook object"""
        nb.metadata.pop('signature', None)
        for cell in self._cells(nb):
            if 'outputs' in cell:
                cell['outputs'] = []
            if 'prompt_number' in cell:
                cell['prompt_number'] = None
        return nb

    def _strip_notebook_output(self, txt):
        """ Strip output cells from an ipython notebook """
        nb = nbformat.reads(txt, nbformat.NO_CONVERT)
        nb = self._strip_output(nb)
        return nbformat.writes(nb)

    def _convert_notebook(self):
        """
    Run a notebook through Jupyter to convert to script.
    This is used as a fallback when directly inserting notebook into
    sql db fails due to images as bytestrings (which fail for columsn of type TEXT in sqlite, for example)
    and failure happens even after stripping output from nb
    """
        assert (self.file_ext == 'ipynb')
        temp_nb = '_temp.ipynb'
        with open(temp_nb, 'w') as f:
            f.write(self.code)
        # going with stdout to avoid issues where jupyter changes file extension (sometimes .py sometimes .txt)
        proc = subprocess.Popen([
            'jupyter', 'nbconvert', temp_nb, '--to', 'script', '--stdout'
        ],
                                stdout=subprocess.PIPE)
        converted, _ = proc.communicate()
        self.code = converted
        os.remove(temp_nb)
        # we relabel language info as plain Python
        self.language = 'Python'
        self.file_ext = 'py'

    def _parse_script_code(self, txt):
        return txt

    def parse_user(self, src):
        r = re.compile('kernel-header__username.*')
        self.user = src.find('a', {'class': r})['href'][1:]
        return self.user

    def parse_votes(self, src):
        r = re.compile('vote-button__vote-count.*')
        self.votes = int(src.find('span', {'class': r}).text)
        return self.votes

    def parse_language(self, src=None):
        r = re.compile('(R?(Python)?) notebook')
        self.language = src.find(text=r).split(' ')[0]
        return self.language

    def _parse_raw_code(self, code_download_link):
        result = wget.download(code_download_link)
        assert (result)
        print "\n"
        temp_file = open(result, 'r')
        code = temp_file.read()
        temp_file.close()
        os.remove(result)
        ext = result.split('.')[-1]
        return ext, code

    def values(self):
        return (self.link, self.user, self.votes, self.language, self.code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle Kernels")
    parser.add_argument('db', type=str, help='Path to sqlite3 database')
    parser.add_argument('table', type=str, help="Name of table to insert into")
    parser.add_argument(
        'col',
        nargs='?',
        type=str,
        help="Link column to avoid downloading repeated notebooks",
        default='link'
    )
    parser.add_argument(
        '-n',
        '--num',
        type=int,
        help="Limited number of scripts to download",
        default=None
    )
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        help="Language for notebooks",
        default='Python'
    )
    parser.add_argument(
        '-s',
        '--sort',
        type=str,
        help="Sort notebooks by criteria",
        default='Most votes'
    )
    parser.add_argument(
        '--populate',
        action='store_true',
        help='Prepopulate crawler with existing links'
    )
    args = parser.parse_args()

    kc = KaggleCrawler(args.language, args.sort)
    if not os.path.exists(args.db):
        raise ValueError("Unknown path for database: %s" % args.db)
    db_conn = sqlite3.connect(args.db)
    if args.populate:
        prepop = kc.populate_downloaded(db_conn, args.table, col=args.col)
        print "Already downloaded %d" % prepop
    # download notebooks
    kc.crawl(db_conn, args.table, n=args.num)
    db_conn.close()
