from typing import Tuple

from pyclick.click_models.task_centric.TaskCentricSearchSession import \
    TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult


class YandexRelPredChallengeParser:
    """
    A parser for the publicly available dataset, released by Yandex (https://www.yandex.com)
    for the Relevance Prediction Challenge (http://imat-relpred.yandex.ru/en).
    """

    @staticmethod
    def parse(sessions_filename, sessions_range: Tuple[int, int]):
        """
        Parses search sessions within a specified range, formatted according to the
        Yandex Relevance Prediction Challenge (RPC).
        Returns a list of SearchSession objects.

        An RPC file contains lines of two formats:
        1. Query action: SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action: SessionID TimePassed TypeOfAction URLID

        :param sessions_filename: The name of the file with search sessions formatted according to RPC.
        :param sessions_range: Tuple of (start_index, end_index) for sessions to parse (0-based, start inclusive, end exclusive - like Python range()).
        :returns: A list of parsed search sessions within the specified range.
        """
        session_begin, session_end = sessions_range
        session_idx = -1  # Start at -1 so first session becomes 0
        sessions = []

        # Track current session state
        current_session = None
        current_task = None
        current_results = None

        with open(sessions_filename, "r") as file:
            for line in file:
                row = line.strip().split("\t")
                is_query = len(row) >= 6 and row[2] == "Q"
                is_click = len(row) == 4 and row[2] == "C"

                if is_query:
                    session_idx += 1

                    if session_idx >= session_end:
                        break

                    if session_begin <= session_idx < session_end:
                        task = row[0]
                        query = row[3]
                        results = row[5:]

                        session = TaskCentricSearchSession(task, query)

                        for result in results:
                            search_result = SearchResult(result, 0)
                            session.web_results.append(search_result)

                        sessions.append(session)

                        # Update current session tracking
                        current_session = session
                        current_task = task
                        current_results = results
                    else:
                        # Reset tracking if we're outside the range
                        current_session = None
                        current_task = None
                        current_results = None

                elif is_click and current_session is not None:
                    if row[0] == current_task:
                        clicked_result = row[3]

                        if clicked_result in current_results:
                            index = current_results.index(clicked_result)
                            current_session.web_results[index].click = 1

        return sessions
