import logging
from decimal import Decimal
from django.db import connections, DEFAULT_DB_ALIAS
from django.db.models import Min, Max, Avg, StdDev, Variance, Count
from modeltree.tree import trees
from . import kmeans    # noqa
from . import agg       # noqa


logger = logging.getLogger(__name__)


AGGREGATION_FUNCTIONS = {
    'min': Min,
    'max': Max,
    'mean': Avg,
    'variance': Variance,
    'stddev': StdDev,
    'count': Count,
    'distinct_count': Count,
}

STATISTICAL_FUNCTIONS = ('median', 'sparsity')


def remove_unsupported_aggregate(query, alias, using=None):
    conn = connections[using or DEFAULT_DB_ALIAS]

    if alias in ('stddev', 'variance') and not conn.features.supports_stddev:
        query.aggregates.pop(alias)
        logger.debug('removing "{0}" aggregation because the '
                     'database does not support it.'.format(alias))
    else:
        aggregate = query.aggregates[alias]

        try:
            conn.ops.check_aggregate_support(aggregate)
        except NotImplementedError:
            logger.debug('removing "{0}" aggregation because the '
                         'database does not support it for the field type'
                         .format(alias))
            query.aggregates.pop(alias)


def distribution(field, context=None, model=None):
    tree = trees[field.model]

    if context:
        queryset = context.apply(tree=tree)
    else:
        queryset = field.model.objects.all()

    value_field_name = field.value_field.name

    queryset = queryset.values(value_field_name)

    if model is None:
        model = field.model

    # The count is the primary key of model of interest,
    # e.g. N occurrences per field value
    count_field = tree.query_string_for_field(model._meta.pk)

    # Annocate with the alias of 'count'
    queryset = queryset.annotate(count=Count(count_field))

    # Returns a list of (value, count) pairs
    return queryset.values_list(value_field_name, 'count')


def summary(field, context=None, functions=None):
    "Returns a dict of statistics about the field."
    tree = trees[field.model]

    if context:
        base_queryset = context.apply(tree=tree)
    else:
        base_queryset = field.model.objects.all()

    if not functions:
        functions = AGGREGATION_FUNCTIONS.keys() + list(STATISTICAL_FUNCTIONS)

    value_field_name = field.value_field.name

    # Manually add aggregates to underlying query since calling
    # QuerySet.aggregate will execute the query, which we do not want.
    # See https://github.com/django/django/blob/master/django/db/models/query.py#L304-L322  # noqa
    queryset = base_queryset.all()

    # Exclude null values
    if field.value_field.null:
        queryset = base_queryset.exclude(**{value_field_name: None})

    query = queryset.query

    for alias in functions:
        if alias in AGGREGATION_FUNCTIONS:
            func = AGGREGATION_FUNCTIONS[alias]

            # Special case for distinct count
            if alias == 'distinct_count':
                expr = func(value_field_name, distinct=True)
            else:
                expr = func(value_field_name)

            query.add_aggregate(expr, query.model, alias, is_summary=True)

            # Immediately check and remove the aggregate if it is unsupported.
            # This uses the database-specific expression that has been added
            # with the above line.
            remove_unsupported_aggregate(query, alias, using=queryset._db)

    # Execute the aggregation
    stats = queryset.aggregate()

    if 'median' in functions:
        values = list(base_queryset.values_list(value_field_name, flat=True)
                      .order_by(field.order_field.name))

        stats['median'] = median(values)

    if 'sparsity' in functions:
        stats['sparsity'] = sparsity(base_queryset, (value_field_name,))

    return stats


def median(values):
    """Calculates the median value of the provided values.

    Note, the values are assumed to be pre-ordered.
    """
    length = len(values)

    if length == 0:
        return

    midpoint = int(length / 2)

    # Take average of two midpoints if a number
    if length % 2 == 0 and isinstance(values[0], (int, float, Decimal)):
        return sum(values[midpoint - 1:midpoint]) / 2.0

    return values[midpoint]


def sparsity(queryset, fields):
    """Calculates the sparsity of data for one or more fields.

    A record must have NULL values for all fields specified to be considered
    a 'sparse' record.
    """
    if not fields:
        raise ValueError('At least one field must be provided')

    count = queryset.count()

    # No data, 100% sparsity
    if count == 0:
        return 1.0

    filters = {}
    template = '{0}__isnull'

    for field in fields:
        filters[template.format(field)] = True

    nulls = queryset.filter(**filters).count()

    return nulls / float(count)
