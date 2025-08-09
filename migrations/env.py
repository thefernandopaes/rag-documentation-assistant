from __future__ import with_statement
import os
from logging.config import fileConfig
import sys

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure project root is on sys.path regardless of CWD
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import create_app, db  # noqa: E402
from config import Config  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = db.metadata


def run_migrations_offline():
    url = Config.get_database_uri()
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    app = create_app()
    with app.app_context():
        configuration = config.get_section(config.config_ini_section)
        configuration["sqlalchemy.url"] = Config.get_database_uri()
        connectable = engine_from_config(
            configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection, target_metadata=target_metadata, compare_type=True
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

