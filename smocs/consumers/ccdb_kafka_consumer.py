"""Write run-averaged calibration predictions from Kafka to a CCDB instance.

Input records use the normal SMOCS envelope and must include ``run_number``.
Prediction records carry the configured value in ``channels``; a ``run_end``
record with the same run number finalizes the average and creates the CCDB
assignment for that single run.
"""

import json
import logging
import os
from typing import Any, Dict, Set, Tuple

from smocs.cores import KafkaConsumerBase
from smocs.utils import ConfigLoader, setup_logging

try:
	import ccdb
except ImportError as error:  # pragma: no cover - depends on deployment image
	ccdb = None
	CCDB_IMPORT_ERROR = error
else:
	CCDB_IMPORT_ERROR = None


class CCDBConsumer(KafkaConsumerBase):
	"""Aggregate one numeric prediction per Kafka event and publish per run."""

	def __init__(self):
		config_path = os.environ.get('CONFIG_PATH', '/app/config.yaml')
		config = ConfigLoader(config_path).config.get('ccdb_consumer', {})
		if not isinstance(config, dict):
			raise ValueError('ccdb_consumer configuration must be a mapping')

		kafka_broker_url = os.environ.get('KAFKA_BROKER_URL', 'kafka-broker:9092')
		topics = self._get_required_list(config, 'kafka_topics')
		group_id = config.get('consumer_group', 'ccdb-calibration-writer-v1')
		super().__init__(kafka_broker_url, group_id, topics)

		if ccdb is None:
			raise RuntimeError(
				'CCDB Python package is unavailable. Install the repository at /app/ccdb '
				'with `pip install /app/ccdb` before starting this consumer.'
			) from CCDB_IMPORT_ERROR

		self.ccdb_connection = self._get_required_environment('CCDB_CONNECTION')
		self.ccdb_user = self._get_required_environment('CCDB_USER')
		self.ccdb_tables = self._get_calibration_tables(config)
		self.table_assignments = self._get_table_assignments(config)
		self.variation = config.get('variation', 'default')
		self.run_end_event_type = config.get('run_end_event_type', 'run_end')
		self.run_aggregates: Dict[Tuple[int, str], Tuple[int, float]] = {}
		self.processed_events: Set[str] = set()
		self.published_runs: Set[Tuple[int, str]] = set()

		self.ccdb_client = None
		self.setup_ccdb_client()

		logging.info(
			'CCDB consumer configured for calibration tables %s, variation %s',
			self.ccdb_tables,
			self.variation,
		)

	def setup_ccdb_client(self) -> None:
		"""Set up and validate the CCDB client."""
		try:
			self.ccdb_client = ccdb.AlchemyProvider()
			self.ccdb_client.connect(self.ccdb_connection)
			self.ccdb_client.authentication.current_user_name = self.ccdb_user
			for value_field, table_config in self.ccdb_tables.items():
				table = self.ccdb_client.get_type_table(table_config['path'])
				column_names = [column.name for column in table.columns]
				if table_config['column'] not in column_names:
					raise ValueError(
						f"CCDB table {table_config['path']} has no {table_config['column']} column for {value_field}"
					)
				if table_config['row'] >= table.rows_count:
					raise ValueError(
						f"CCDB table {table_config['path']} has no row {table_config['row']} for {value_field}"
					)
				if table.rows_count != 1:
					raise ValueError(
						f"CCDB table {table_config['path']} has {table.rows_count} rows; only one-row tables are supported"
					)
				unknown_columns = set(table_config['values']) - set(column_names)
				if unknown_columns:
					raise ValueError(
						f"CCDB table {table_config['path']} has no configured value columns {sorted(unknown_columns)}"
					)
				missing_columns = set(column_names) - {table_config['column']} - set(table_config['values'])
				if missing_columns:
					raise ValueError(
						f"CCDB table {table_config['path']} is missing configured values for {sorted(missing_columns)}"
					)
			for assignment_name, assignment_config in self.table_assignments.items():
				table = self.ccdb_client.get_type_table(assignment_config['path'])
				if len(assignment_config['rows']) != table.rows_count:
					raise ValueError(f"CCDB table {assignment_config['path']} row count does not match {assignment_name}")
				if any(len(row) != table.columns_count for row in assignment_config['rows']):
					raise ValueError(f"CCDB table {assignment_config['path']} column count does not match {assignment_name}")
			logging.info('CCDB client connected successfully')
		except Exception as error:
			logging.error('Failed to set up CCDB client: %s', error)
			raise

	@staticmethod
	def _get_required_environment(name: str) -> str:
		value = os.environ.get(name)
		if not value:
			raise ValueError(f'{name} environment variable is required')
		return value

	@staticmethod
	def _get_required_list(config: Dict[str, Any], name: str) -> list:
		value = config.get(name)
		if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
			raise ValueError(f'ccdb_consumer.{name} must be a non-empty list of strings')
		return value

	@staticmethod
	def _get_calibration_tables(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		value = config.get('calibration_tables')
		if not isinstance(value, dict) or not value:
			raise ValueError('ccdb_consumer.calibration_tables must be a non-empty mapping')

		calibration_tables = {}
		for value_field, table_config in value.items():
			if not isinstance(value_field, str) or not value_field or not isinstance(table_config, dict):
				raise ValueError('ccdb_consumer.calibration_tables must map fields to table configurations')

			table_path = table_config.get('path')
			column_name = table_config.get('column')
			row_index = table_config.get('row', 0)
			values = table_config.get('values', {})
			if not isinstance(table_path, str) or not table_path.startswith('/'):
				raise ValueError(f'CCDB table path for {value_field} must be absolute')
			if not isinstance(column_name, str) or not column_name:
				raise ValueError(f'CCDB column for {value_field} must be configured')
			if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
				raise ValueError(f'CCDB row for {value_field} must be a non-negative integer')
			if not isinstance(values, dict) or not all(isinstance(name, str) and name for name in values):
				raise ValueError(f'CCDB values for {value_field} must map column names to values')
			if column_name in values:
				raise ValueError(f'CCDB values for {value_field} must not override target column {column_name}')

			calibration_tables[value_field] = {
				'path': table_path,
				'column': column_name,
				'row': row_index,
				'values': values,
			}
		return calibration_tables

	@staticmethod
	def _get_table_assignments(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		value = config.get('table_assignments', {})
		if not isinstance(value, dict):
			raise ValueError('ccdb_consumer.table_assignments must be a mapping')

		assignments = {}
		for assignment_name, assignment_config in value.items():
			if not isinstance(assignment_name, str) or not isinstance(assignment_config, dict):
				raise ValueError('ccdb_consumer.table_assignments must map names to configurations')
			path = assignment_config.get('path')
			rows = assignment_config.get('rows')
			if not isinstance(path, str) or not path.startswith('/') or not isinstance(rows, list) or not rows:
				raise ValueError(f'Invalid table assignment configuration for {assignment_name}')
			if not all(isinstance(row, list) and row for row in rows):
				raise ValueError(f'CCDB table assignment {assignment_name} must contain non-empty rows')
			assignments[assignment_name] = {
				'path': path,
				'variation': assignment_config.get('variation', config.get('variation', 'default')),
				'rows': rows,
			}
		return assignments

	@staticmethod
	def _prediction_fields(value: Any) -> Set[str]:
		if isinstance(value, str):
			return {value}
		if isinstance(value, dict) and set(value) == {'negate'} and isinstance(value['negate'], str):
			return {value['negate']}
		return set()

	def _configured_prediction_fields(self) -> Set[str]:
		fields = set(self.ccdb_tables)
		for assignment_config in self.table_assignments.values():
			for row in assignment_config['rows']:
				for value in row:
					fields.update(self._prediction_fields(value))
		return fields

	@staticmethod
	def parse_message_data(message: Any) -> Dict[str, Any]:
		if isinstance(message, bytes):
			message = message.decode('utf-8')
		data = json.loads(message)
		if not isinstance(data, dict):
			raise ValueError('Kafka message must decode to an object')
		return data

	@staticmethod
	def _parse_run_number(message_data: Dict[str, Any]) -> int:
		value = message_data.get('run_number')
		if isinstance(value, bool):
			raise ValueError('run_number must be a non-negative integer')
		try:
			run_number = int(value)
		except (TypeError, ValueError) as error:
			raise ValueError('Kafka message must contain an integer run_number') from error
		if run_number < 0 or str(run_number) != str(value).strip():
			raise ValueError('run_number must be a non-negative integer')
		return run_number

	def _extract_predictions(self, message_data: Dict[str, Any]) -> Dict[str, float]:
		channels = message_data.get('channels', {})
		inference_outputs = message_data.get('inference_result', {}).get('predicted_outputs', {})
		predictions = {}
		for value_field in self._configured_prediction_fields():
			value = channels.get(value_field, inference_outputs.get(value_field))
			if value is None:
				continue
			if isinstance(value, bool):
				raise ValueError(f'{value_field} must be numeric')
			try:
				prediction = float(value)
			except (TypeError, ValueError) as error:
				raise ValueError(f'{value_field} must be numeric') from error
			if prediction != prediction or prediction in (float('inf'), float('-inf')):
				raise ValueError(f'{value_field} must be finite')
			predictions[value_field] = prediction
		return predictions

	@staticmethod
	def _event_id(topic: str, partition: int, offset: int) -> str:
		return f'{topic}:{partition}:{offset}'

	def _record_predictions(self, event_id: str, run_number: int, predictions: Dict[str, float]) -> None:
		if event_id in self.processed_events:
			return
		self.processed_events.add(event_id)
		for value_field, prediction in predictions.items():
			key = (run_number, value_field)
			sample_count, value_sum = self.run_aggregates.get(key, (0, 0.0))
			self.run_aggregates[key] = (sample_count + 1, value_sum + prediction)

	def write_ccdb_assignments(self, event_id: str, run_number: int) -> None:
		if event_id in self.processed_events:
			return
		self.processed_events.add(event_id)

		for value_field, table_config in self.ccdb_tables.items():
			key = (run_number, value_field)
			if key in self.published_runs:
				logging.info('Run %s field %s was already published', run_number, value_field)
				continue

			aggregate = self.run_aggregates.get(key)
			if aggregate is None:
				continue

			sample_count, value_sum = aggregate
			mean = value_sum / sample_count
			calibration_path = table_config['path']
			table = self.ccdb_client.get_type_table(calibration_path)
			data = [[table_config['values'].get(column.name) for column in table.columns]]
			column_index = [column.name for column in table.columns].index(table_config['column'])
			data[table_config['row']][column_index] = mean
			comment = json.dumps(
				{
					'producer': 'smocs.ccdb_kafka_consumer',
					'run_number': run_number,
					'field': value_field,
					'column': table_config['column'],
					'sample_count': sample_count,
				},
				sort_keys=True,
			)

			assignment = self.ccdb_client.create_assignment(
				data=data,
				path=calibration_path,
				min_run=run_number,
				max_run=run_number,
				variation_name=self.variation,
				comment=comment,
			)
			self.published_runs.add(key)
			logging.info(
				'Published run %s %s average %.12g from %s samples as CCDB assignment %s',
				run_number,
				value_field,
				mean,
				sample_count,
				assignment.id,
			)

		for assignment_name, assignment_config in self.table_assignments.items():
			key = (run_number, assignment_name)
			if key in self.published_runs:
				continue
			prediction_values = {}
			for value_field in self._configured_prediction_fields():
				aggregate = self.run_aggregates.get((run_number, value_field))
				if aggregate is not None:
					prediction_values[value_field] = aggregate[1] / aggregate[0]

			try:
				data = [
					[self._resolve_assignment_value(value, prediction_values) for value in row]
					for row in assignment_config['rows']
				]
			except KeyError as error:
				logging.warning('Run %s has no prediction for %s; skipping %s', run_number, error.args[0], assignment_name)
				continue

			assignment = self.ccdb_client.create_assignment(
				data=data,
				path=assignment_config['path'],
				min_run=run_number,
				max_run=run_number,
				variation_name=assignment_config['variation'],
				comment=f'Online update based on EPICS; run-averaged SMOCS predictions for {assignment_name}',
			)
			self.published_runs.add(key)
			logging.info('Published run %s CCDB assignment %s as %s', run_number, assignment.id, assignment_name)

	@staticmethod
	def _resolve_assignment_value(value: Any, prediction_values: Dict[str, float]) -> Any:
		if isinstance(value, str):
			return prediction_values[value]
		if isinstance(value, dict) and set(value) == {'negate'}:
			return -prediction_values[value['negate']]
		return value

	def process_message(self, message, topic, partition, offset) -> bool:
		"""Record a prediction or publish its run average on a run-end event."""
		try:
			message_data = self.parse_message_data(message)
			run_number = self._parse_run_number(message_data)
			event_id = self._event_id(topic, partition, offset)

			if message_data.get('event_type') == self.run_end_event_type:
				self.write_ccdb_assignments(event_id, run_number)
			else:
				self._record_predictions(event_id, run_number, self._extract_predictions(message_data))
			return True
		except (TypeError, ValueError, json.JSONDecodeError) as error:
			logging.error('Rejected %s:%s:%s: %s', topic, partition, offset, error)
			return False
		except Exception:
			logging.exception('Failed to process %s:%s:%s', topic, partition, offset)
			return False

	def cleanup(self) -> None:
		if getattr(self, 'ccdb_client', None) and self.ccdb_client.is_connected:
			self.ccdb_client.disconnect()
		super().cleanup()


def main() -> None:
	setup_logging()
	logging.info('Starting CCDB consumer...')
	CCDBConsumer().start()


if __name__ == '__main__':
	main()


# Compatibility alias for existing imports while callers migrate to CCDBConsumer.
CCDBKafkaConsumer = CCDBConsumer
