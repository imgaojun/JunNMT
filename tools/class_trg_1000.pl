#!/usr/bin/perl

use strict;
use warnings;

my $rnn_dir = "/home/dhan/word_class/toolkit/class_trg_1000";

my $cmd;

my $train_data_dir = "/home/dhan/word_class/train";
my $test_data_dir = "/home/dhan/word_class/test";

my $src_lang = "cn";
my $tgt_lang = "en";


my $work_dir = "/home/dhan/word_class/rnn/class_trg_1000";
my $data_dir = "$work_dir/data";
my $config_file = "$work_dir/config.ini";
run("mkdir -p $work_dir") unless -d $work_dir;
run("mkdir -p $data_dir") unless -d $data_dir;

my @data_set = ("nist02", "nist03", "nist04", "nist05", "nist06", "nist08");
my $tune_set = "nist06";
my $test_set = "nist03";

if (0){
	
    $cmd = "python $rnn_dir/shuffle.py $train_data_dir/train.$src_lang $train_data_dir/train.$tgt_lang";
    run("$cmd");
    run("ln -s $train_data_dir/train.$src_lang.shuffle $data_dir/ch.txt.shuffle");
    run("ln -s $train_data_dir/train.$tgt_lang.shuffle $data_dir/en.txt.shuffle");
}



#step 0: prepare data
if (0) {
  foreach my $set (@data_set) {
    $cmd = "$rnn_dir/data/plain2sgm src $test_data_dir/$set.$src_lang > $data_dir/$set.$src_lang.sgm";
    run("$cmd");
    $cmd = "$rnn_dir/data/plain2sgm ref $test_data_dir/$set.$tgt_lang"."0 $test_data_dir/$set.$tgt_lang"."1 $test_data_dir/$set.$tgt_lang"."2 $test_data_dir/$set.$tgt_lang"."3 > $data_dir/$set.$tgt_lang.sgm";
    run("$cmd");
    $cmd = "ln -s $test_data_dir/$set.$src_lang $data_dir/$set.$src_lang ";
    run("$cmd");
    for (my $i = 0; $i < 4; $i++) {
      $cmd = "ln -s $test_data_dir/$set.$tgt_lang$i $data_dir/$set.$tgt_lang$i";
      run("$cmd");
    }
  }

  $cmd = "ln -s $test_data_dir/$tune_set.$src_lang $data_dir/valid_src";
  run("$cmd");
  $cmd = "ln -s $data_dir/$tune_set.$src_lang.sgm $data_dir/valid_src.sgm";
  run("$cmd");
  $cmd = "ln -s $data_dir/$tune_set.$tgt_lang.sgm $data_dir/valid_trg.sgm";
  run("$cmd");

  $cmd = "ln -s $data_dir/$test_set.$src_lang.sgm $data_dir/test_src.sgm";
  run("$cmd");
  $cmd = "ln -s $test_data_dir/$test_set.$src_lang $data_dir/test_src";
  run("$cmd");
  $cmd = "ln -s $data_dir/$test_set.$tgt_lang.sgm $data_dir/test_trg.sgm";
  run("$cmd");

  $cmd = "ln -s $rnn_dir/data/plain2sgm $data_dir/plain2sgm";
  run("$cmd");
  $cmd = "ln -s $rnn_dir/data/mteval-v11b.pl $data_dir/mteval-v11b.pl";
  run("$cmd");
  $cmd = "ln -s $rnn_dir/data/multi-bleu.perl $data_dir/multi-bleu.perl";
  run("$cmd");
}

#step 2: run prepare data
if (0) {	
  my $prepare_log = "$work_dir/prepare.log";
  $cmd = "python $rnn_dir/prepare_data.py --ini $config_file 1> $work_dir/prepare.log 2>&1";
  run("$cmd");
}

#step 3: train
if (1) {
  $cmd = "nohup python $rnn_dir/train.py --ini $config_file 1> $work_dir/train.log 2>&1 &";
  run("$cmd");
}


# grep "valid_bleu" workspace/train.log
# grep "valid_bleu" workspace/train.log  | sort -k7nr | head

#step 4: test a specified model (or multiple models)
if (0) {
  my $best_iter = "166000";
  my @vec_model_files = <$work_dir/$best_iter\/train_model.npz>;
  foreach my $model_file (@vec_model_files) {
    print "model file: $model_file\n";
    my $dir_name;
    my $base_name;
    ($dir_name, $base_name) = $model_file =~ m|^(.*[/\\])([^/\\]+?)$|;
    #my $dir_name = dirname($model_file);
    #my $base_name = basename($model_file);

   foreach my $set(@data_set) {
     # if (-e "$dir_name/$set.tran.eval") {
     #   next;
     # }
     run("rm $dir_name/config.ini") if -e "$dir_name/config.ini";
      $cmd = "cp $config_file $dir_name/config.ini; echo \"saveto_best=$model_file\" >> $dir_name/config.ini";
      run("$cmd");
      $cmd = "python $rnn_dir/sampling.py --ini $dir_name/config.ini $data_dir/$set.$src_lang $data_dir/$set.$tgt_lang $dir_name/$set.tran $dir_name/$set.align >1"; 
      run("$cmd");

      $cmd = "perl $rnn_dir/data/multi-bleu.perl $data_dir/$set.$tgt_lang < $dir_name/$set.tran > $dir_name/$set.tran.eval";
      run("$cmd");
      $cmd = "/data1/dhan/mt-exp/toolkit/cdec-2014-10-12/mteval/fast_score -r $data_dir/$set.$tgt_lang"."0 -r $data_dir/$set.$tgt_lang"."1 -r $data_dir/$set.$tgt_lang"."2 -r $data_dir/$set.$tgt_lang"."3 -i $dir_name/$set.tran 1> $dir_name/$set.tran.evalibm 2>&1";
      run("$cmd");

     $cmd = "$rnn_dir/data/plain2sgm tst $dir_name/$set.tran > $dir_name/$set.tran.sgm";
      run("$cmd");
      $cmd = "perl $rnn_dir/data/mteval-v11b.pl -r $data_dir/$set.$tgt_lang.sgm -s $data_dir/$set.$src_lang.sgm -t $dir_name/$set.tran.sgm > $dir_name/$set.tran.eval11b";
      run("$cmd");
    }
  }
}


#step 6: force alignment
my $align_data_dir = "/data1/dhan/mt-exp/align_data";
if (0)  {
  my $best_iter = "157000";
  my @vec_model_files = <$work_dir/$best_iter\/train_model.npz>;
  foreach my $model_file (@vec_model_files) {
    print "model file: $model_file\n";
    my $dir_name;
    my $base_name;
    ($dir_name, $base_name) = $model_file =~ m|^(.*[/\\])([^/\\]+?)$|;
    $cmd = "python $rnn_dir/aligner.py --ini $dir_name/config.ini --model $model_file $align_data_dir/900.cn $align_data_dir/900.en $dir_name/900.align 1> $dir_name/900.log 2>&1";
    run("$cmd");
    $cmd = "python $align_data_dir/cal_alignment_accuracy.py $dir_name/900.align $align_data_dir/900.align > $dir_name/900.align.eval";
    run("$cmd");
  }
}

sub run {
  my $mycmd = shift;
  print "$mycmd\n";
  system("$mycmd");
}

