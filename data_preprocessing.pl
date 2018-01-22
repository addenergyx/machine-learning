#! /usr/bin/perl

use strict;
use warnings;
use RNA;

#This script is to prepare the dataset for the neural network. Andrew suggested other features that they don't currently record such as number of a specific nucleotide and g-c content. This code will take the sequence from each observation to get the features required.

my $sequence = "CAGAACCGGAGGACAAAGTACAAACGGCAGAAGCTGGAGGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGCATTGCCACGAAGCAGGCCAATGGGGAGGACATCGATGTCACCTCCAATGACTAGGGTGGGCAACCACAAACCCACGAGGGCAGAGTGCTGCTTGCTGCTGGCCAGGCCCCTGCGTGGGCCCAAGCTGGACTCTGGCCACTCCCTGGCCAGGCTTTGGGGAGGCCTGGAGTCATG";
print "\nSequence:\n$sequence\n\n";

#Nucleotide count

#Using () to change the the count from scalar to list context
#my $a_count = ($sequence =~ tr/a//i); <- doesn't work

my $a_count = () = $sequence =~ /a/ig; 
my $c_count = () = $sequence =~ /c/ig; 
my $t_count = () = $sequence =~ /t/ig; 
my $g_count = () = $sequence =~ /g/ig; 
print "A : $a_count \nC : $a_count \nT : $t_count \nG : $g_count \n";

#G-C bonds are stronger then A-T bonds as they have an extra hydrogen bond, therefore making them harder to break
my $gc_content = () = $sequence =~ /gc/ig;
print "G-C Content : $gc_content\n"; 

#3 or more T bases in a row can pervent polymerase from working properlly and therefore it can drop off
my $triple_t_count = () = $sequence =~ /ttt/ig;
print "TTT Content : $triple_t_count\n"; 

#Later should look into using a hash containing these variables for each sequence
#my $string = q<ABCDABDIDAOFOOFAA>;
#my %count;
#for (split(//, $string)){$count{$_}++;}
#print(q<Matches >, $count{A}, "\n");

#Secondary structure: The lab uses RNAfold WebServer to get information on the secondary structure of the DNA/RNA sequence. ViennaRNA has a package I can install and use to get the features needed

my ($structure, $mfe) = RNA::fold($sequence);
print "Minimum free energy prediction: $mfe\n\n"
 




