import json

from smocs.consumers.ccdb_kafka_consumer import CCDBConsumer


class FakeAssignment:
    id = 17


class FakeColumn:
    def __init__(self, name):
        self.name = name


class FakeTable:
    columns = [
        FakeColumn('FDC_ADC_ASCALE'),
        FakeColumn('FDC_ADC_TSCALE'),
        FakeColumn('FDC_TDC_SCALE'),
    ]
    rows_count = 1


class FakeCDCDriftParametersTable:
    columns = [FakeColumn(f'column_{index}') for index in range(11)]
    columns_count = 11
    rows_count = 2


class FakeCDCDigiScalesTable:
    columns = [FakeColumn('CDC_ADC_ASCALE'), FakeColumn('CDC_ADC_TSCALE')]
    columns_count = 2
    rows_count = 1


class FakeCCDBClient:
    def __init__(self):
        self.calls = []

    def create_assignment(self, **kwargs):
        self.calls.append(kwargs)
        return FakeAssignment()

    def get_type_table(self, path):
        if path == '/CDC/drift_parameters':
            return FakeCDCDriftParametersTable()
        if path == '/CDC/digi_scales':
            return FakeCDCDigiScalesTable()
        return FakeTable()


def make_consumer():
    consumer = object.__new__(CCDBConsumer)
    consumer.run_end_event_type = 'run_end'
    consumer.ccdb_tables = {}
    consumer.variation = 'default'
    consumer.table_assignments = {
        'fdc_digi_scales': {
            'path': '/test/fdc/gain',
            'variation': 'ceac',
            'rows': [['predicted_fdc_gain', 0.8, 0.115]],
        }
    }
    consumer.run_aggregates = {}
    consumer.processed_events = set()
    consumer.published_runs = set()
    consumer.ccdb_client = FakeCCDBClient()
    return consumer


def test_consumer_averages_predictions_and_writes_one_run_assignment():
    consumer = make_consumer()

    first_prediction = {
        'timestamp': 1.0,
        'run_number': 8142,
        'channels': {'predicted_fdc_gain': 1.0},
    }
    second_prediction = {
        'timestamp': 2.0,
        'run_number': 8142,
        'channels': {'predicted_fdc_gain': 3.0},
    }
    run_end = {
        'timestamp': 3.0,
        'run_number': 8142,
        'event_type': 'run_end',
        'channels': {},
    }

    assert consumer.process_message(json.dumps(first_prediction), 'predictions', 0, 1)
    assert consumer.process_message(json.dumps(second_prediction), 'predictions', 0, 2)
    assert consumer.process_message(json.dumps(run_end), 'predictions', 0, 3)

    assert len(consumer.ccdb_client.calls) == 1
    assert consumer.ccdb_client.calls[0]['data'] == [[2.0, 0.8, 0.115]]
    assert consumer.ccdb_client.calls[0]['path'] == '/test/fdc/gain'
    assert consumer.ccdb_client.calls[0]['min_run'] == 8142
    assert consumer.ccdb_client.calls[0]['max_run'] == 8142
    assert consumer.ccdb_client.calls[0]['variation_name'] == 'ceac'


def test_consumer_routes_each_configured_prediction_to_its_table():
    consumer = make_consumer()
    consumer.table_assignments['fdc_digi_scales_offset'] = {
        'path': '/test/fdc/offset',
        'variation': 'ceac',
        'rows': [[0.333, 'predicted_fdc_offset', 0.115]],
    }
    prediction = {
        'run_number': 8,
        'channels': {'predicted_fdc_gain': 2.0, 'predicted_fdc_offset': -0.5},
    }
    run_end = {'run_number': 8, 'event_type': 'run_end', 'channels': {}}

    assert consumer.process_message(json.dumps(prediction), 'predictions', 0, 1)
    assert consumer.process_message(json.dumps(run_end), 'predictions', 0, 2)

    assert [call['path'] for call in consumer.ccdb_client.calls] == ['/test/fdc/gain', '/test/fdc/offset']


def test_consumer_ignores_a_replayed_prediction_and_run_end():
    consumer = make_consumer()
    prediction = {
        'timestamp': 1.0,
        'run_number': 4,
        'channels': {'predicted_fdc_gain': 5.0},
    }
    run_end = {
        'timestamp': 2.0,
        'run_number': 4,
        'event_type': 'run_end',
        'channels': {},
    }

    assert consumer.process_message(json.dumps(prediction), 'predictions', 0, 1)
    assert consumer.process_message(json.dumps(prediction), 'predictions', 0, 1)
    assert consumer.process_message(json.dumps(run_end), 'predictions', 0, 2)
    assert consumer.process_message(json.dumps(run_end), 'predictions', 0, 2)

    assert len(consumer.ccdb_client.calls) == 1
    assert consumer.ccdb_client.calls[0]['data'] == [[5.0, 0.8, 0.115]]


def test_consumer_writes_cdc_drift_parameters_from_averaged_predictions():
    consumer = make_consumer()
    consumer.table_assignments = {
        **consumer.table_assignments,
        'cdc_drift_parameters': {
            'path': '/CDC/drift_parameters',
            'variation': 'default',
            'rows': [
                ['a1', 'a2', 0, 'b1', 'b2', 0, 'c1', 'c2', 0, 1.1, -0.08],
                ['a1', {'negate': 'a2'}, 0, 'b1', {'negate': 'b2'}, 0, 'c1', {'negate': 'c2'}, 0, 1.1, -0.08],
            ],
        }
    }
    prediction = {
        'run_number': 9,
        'channels': {'a1': 1.0, 'a2': 2.0, 'b1': 3.0, 'b2': 4.0, 'c1': 5.0, 'c2': 6.0},
    }
    run_end = {'run_number': 9, 'event_type': 'run_end', 'channels': {}}

    assert consumer.process_message(json.dumps(prediction), 'predictions', 0, 1)
    assert consumer.process_message(json.dumps(run_end), 'predictions', 0, 2)

    cdc_call = next(call for call in consumer.ccdb_client.calls if call['path'] == '/CDC/drift_parameters')
    assert cdc_call['data'] == [[1.0, 2.0, 0, 3.0, 4.0, 0, 5.0, 6.0, 0, 1.1, -0.08], [1.0, -2.0, 0, 3.0, -4.0, 0, 5.0, -6.0, 0, 1.1, -0.08]]


def test_consumer_writes_gp_average_to_cdc_digi_scales():
    consumer = make_consumer()
    consumer.table_assignments = {
        **consumer.table_assignments,
        'cdc_digi_scales': {
            'path': '/CDC/digi_scales',
            'variation': 'ceac',
            'rows': [['predicted_cdc_adc_ascale', 0.8]],
        }
    }
    first_prediction = {'run_number': 10, 'channels': {'predicted_cdc_adc_ascale': 0.13}}
    second_prediction = {'run_number': 10, 'channels': {'predicted_cdc_adc_ascale': 0.14}}
    run_end = {'run_number': 10, 'event_type': 'run_end', 'channels': {}}

    assert consumer.process_message(json.dumps(first_prediction), 'gp1-predictions', 0, 1)
    assert consumer.process_message(json.dumps(second_prediction), 'gp1-predictions', 0, 2)
    assert consumer.process_message(json.dumps(run_end), 'gp1-predictions', 0, 3)

    cdc_call = next(call for call in consumer.ccdb_client.calls if call['path'] == '/CDC/digi_scales')
    assert cdc_call['variation_name'] == 'ceac'
    assert cdc_call['data'] == [[0.135, 0.8]]
