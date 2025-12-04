#!/bin/bash
javac -d bin $(find src -name "*.java")
java -cp bin P4 < input.txt
