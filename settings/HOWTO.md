# Advanced techniques with settings
Here's some tips on improving workflow and getting advanced results using layered settings files.
The settings files in this directory are examples of some of these techniques, detailed below:

## Tip: Copy the original settings.json and use it as a basis for any new image
It is recommended that you do not modify the original settings, just so you have a file you can look at to get an idea of a reasonable starting point.

## Multiple settings files explained
A key difference between Prog Rock and Disco is that you can use multiple, partial settings files and they will be layered on top of one another.
The files are loaded in order of how they are specified, and only the settings in the file are applied, overwriting any setting from any previous files.
settings.json in the main directory is always loaded! Another reason to leave it alone! Other settings files simply take precedent.

In practical terms, this means you can have partial settings files that only adjust the variables you want.

### Multiple settings example:
Let's say you have "myscene.json" which contains your lovely prompt, the number of images you want to make, and a few other tweaks.
It's working well, but now you want to take it to the next level and run it at a much higher resolution with more advanced models enabled.
Instead of tweaking the myscene.json to achieve this, you can simply layer on another settings file that has your high-quality preferred settings.

```
python3 prd.py -s settings/myscene.json -s settings/ultra_quality.json
```

This takes all of your "myscene" settings, but then overrides some of them with the higher quality options in ultra_quality.json

## Randomize settings:
Some settings can be randomized, as a way to explore what is possible.
You can see all the settings that work this way in random.json - give it a try!

## Randomizers in your prompt:
Any text in a prompt surrounded by _'s will be replaced by a random line from the corresponding text file.
For example:
```
A _adjective_ _style_ of a _subject_, by _artist_, _site_
```
Becomes something like:
```
A psychotic 3D VR painting of a shop, by Wayne Barlowe, trending on 500px
```
Just make sure the file with your replacements exists! Some have been included but you can add whatever ones you like.

## Changing prompts:
One incredibly cool trick is changing your prompt at a given step. This allows you to initialize your image with one prompt, but then switch to something else that provides the rest of the detail and texture to the scene.
See change_prompt_sample.json for an idea of how this works.
