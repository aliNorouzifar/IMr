'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
__doc__ = """
The ``pm4py.discovery`` module contains the process discovery algorithms implemented in ``pm4py``
"""

from typing import Dict, Set, Optional, Any, Union
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
import pandas as pd
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.util import constants
import warnings



def discover_declare(log: Union[EventLog, pd.DataFrame], allowed_templates: Optional[Set[str]] = None, considered_activities: Optional[Set[str]] = None, min_support_ratio: Optional[float] = None, min_confidence_ratio: Optional[float] = None, activity_key: str = "concept:name", timestamp_key: str = "time:timestamp", case_id_key: str = "case:concept:name") -> Dict[str, Dict[Any, Dict[str, int]]]:
    """
    Discovers a DECLARE model from an event log.

    Reference paper:
    F. M. Maggi, A. J. Mooij and W. M. P. van der Aalst, "User-guided discovery of declarative process models," 2011 IEEE Symposium on Computational Intelligence and Data Mining (CIDM), Paris, France, 2011, pp. 192-199, doi: 10.1109/CIDM.2011.5949297.

    :param log: event log / Pandas dataframe
    :param allowed_templates: (optional) collection of templates to consider for the discovery
    :param considered_activities: (optional) collection of activities to consider for the discovery
    :param min_support_ratio: (optional, decided automatically otherwise) minimum percentage of cases (over the entire set of cases of the log) for which the discovered rules apply
    :param min_confidence_ratio: (optional, decided automatically otherwise) minimum percentage of cases (over the rule's support) for which the discovered rules are valid
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :rtype: ``Dict[str, Any]``

    .. code-block:: python3

        import pm4py

        declare_model = pm4py.discover_declare(log)
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception(
            "the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)

    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(
            log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

    properties = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)
    properties["allowed_templates"] = allowed_templates
    properties["considered_activities"] = considered_activities
    properties["min_support_ratio"] = min_support_ratio
    properties["min_confidence_ratio"] = min_confidence_ratio

    from local_pm4py.declare import algorithm as declare_discovery
    return declare_discovery.apply(log, parameters=properties)


def get_properties(log, activity_key: str = "concept:name", timestamp_key: str = "time:timestamp", case_id_key: str = "case:concept:name", resource_key: str = "org:resource", group_key: Optional[str] = None, **kwargs):
    """
    Gets the properties from a log object

    :param log: Log object
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :param resource_key: (if provided) attribute to be used as resource
    :param group_key: (if provided) attribute to be used as group identifier
    :rtype: ``Dict``
    """
    __event_log_deprecation_warning(log)

    if type(log) not in [pd.DataFrame, EventLog, EventStream]: return {}

    from copy import copy
    parameters = copy(log.properties) if hasattr(log, 'properties') else copy(log.attrs) if hasattr(log,
                                                                                                    'attrs') else {}

    if activity_key is not None:
        parameters[constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = activity_key
        parameters[constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY] = activity_key

    if timestamp_key is not None:
        parameters[constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] = timestamp_key

    if case_id_key is not None:
        parameters[constants.PARAMETER_CONSTANT_CASEID_KEY] = case_id_key

    if resource_key is not None:
        parameters[constants.PARAMETER_CONSTANT_RESOURCE_KEY] = resource_key

    if group_key is not None:
        parameters[constants.PARAMETER_CONSTANT_GROUP_KEY] = group_key

    for k, v in kwargs.items():
        parameters[k] = v

    return parameters




def __event_log_deprecation_warning(log):
    if constants.SHOW_EVENT_LOG_DEPRECATION and not hasattr(log, "deprecation_warning_shown"):
        if constants.SHOW_INTERNAL_WARNINGS:
            if isinstance(log, EventLog) or isinstance(log, Trace):
                warnings.warn("the EventLog class has been deprecated and will be removed in a future release.")
                log.deprecation_warning_shown = True
            elif isinstance(log, Trace):
                warnings.warn("the Trace class has been deprecated and will be removed in a future release.")
                log.deprecation_warning_shown = True
            elif isinstance(log, EventStream):
                warnings.warn("the EventStream class has been deprecated and will be removed in a future release.")
                log.deprecation_warning_shown = True

