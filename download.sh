#!/bin/bash

if [ "$1" = "" -o "$1" = "-h" -o "$1" = "--help" ]; then
  echo "Usage: ./download.sh <dataset> [<dataset> ...]"
  echo ""
  echo "Script for downloading image datasets"
  echo ""
  echo "Positional arguments:"
  echo "  <dataset> [<dataset> ...]"
  echo "                        Target dataset to dowload. Supported datasets: coco2014train"
  echo "                        coco2014test, coco2017train, coco2017test, wikiart"
else
  basedir=data
  mkdir -p $basedir
  for var in "$@"; do
    case $var in
    "coco2014train")
      wget -O $basedir/coco2014train.zip http://images.cocodataset.org/zips/train2014.zip
      unzip $basedir/coco2014train.zip -d $basedir/coco2014train
      rm $basedir/coco2014train.zip
      ;;
    "coco2014test")
      wget -O $basedir/coco2014test.zip http://images.cocodataset.org/zips/test2014.zip
      unzip $basedir/coco2014test.zip -d $basedir/coco2014test
      rm $basedir/coco2014test.zip
      ;;
    "coco2017train")
      wget -O $basedir/coco2017train.zip http://images.cocodataset.org/zips/train2017.zip
      unzip $basedir/coco2017train.zip -d $basedir/coco2017train
      rm $basedir/coco2017train.zip
      ;;
    "coco2017test")
      wget -O $basedir/coco2017test.zip http://images.cocodataset.org/zips/test2017.zip
      unzip $basedir/coco2017test.zip -d $basedir/coco2017test
      rm $basedir/coco2017test.zip
      ;;
    "wikiart")
      wget -O $basedir/wikiart.zip http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
      unzip $basedir/wikiart.zip -d $basedir/wikiart
      rm $basedir/wikiart.zip
      ;;
    *)
      echo "$var not supported"
      ;;
    esac
  done
fi
