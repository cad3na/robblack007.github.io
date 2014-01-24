help:
	@echo "You may provide several parameters, like:"
	@echo "make [target] KEY=\"value\""
	@echo ""
	@echo "You may provide the TOPIC variable to the 'new' target."
	@echo ""
	@echo "Your targets may be:"
	@echo "  help - Shows this message."
	@echo "  clean - Removes all the files from _site directory."
	@echo "  build - Builds the site and notificates you when it's done."
	@echo "  watch - Serves the site (watching for changes) in a local host and notificates you with the URL."
	@echo "  review - Serves the site in a local host and notificates you with the URL."
	@echo "  new - Creates a new post."

.PHONY: clean

clean:
	rm -rf _site/*

build: clean
	bundle exec jekyll build
	chmod -R 755 _site/*
	terminal-notifier -message "Blog built." -title $(SITE) -sound default

watch: clean
	bundle exec jekyll serve --watch

review: clean
	bundle exec jekyll serve

new:
	echo "---" >> $(FILE)
	echo "title: $(TOPIC)" >> $(FILE)
	echo "layout: post" >> $(FILE)
	echo "category: articles" >> $(FILE)
	echo "tags: [tag1, tag2]" >> $(FILE)
	echo "image:" >> $(FILE)
	echo "  feature: file.png" >> $(FILE)
	echo "  credit: Roberto Cadena Vega" >> $(FILE)
	echo "comments: true" >> $(FILE)
	echo "---" >> $(FILE)
	open $(FILE)

LOCAL = "http://localhost:4000"
SITE = "http://robblack007.github.io"

TOPIC ?= new article
FILE = $(shell date "+./_posts/%Y-%m-%d-$(TOPIC).md" | sed -e y/\ /-/)