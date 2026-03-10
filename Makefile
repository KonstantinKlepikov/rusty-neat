# target: all - Default target. Does nothing.
all:
	echo "Hello, this is make for tiny-rewards-tg"
	echo "Try 'make help' and search available options"

# target: help - List of options
help:
	egrep "^# target:" [Mm]akefile

# target: check - run cargo checks
check:
	cargo fmt --all; cargo test
