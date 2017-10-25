package comp9313.ass4;

import java.io.DataInput;
import java.util.ArrayList;

import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SetSimJoin {
    // a writable pair store two integers
	public static class IntPair implements WritableComparable<IntPair>{
		private int first, second;
		
		public IntPair() {
		}
		
		public IntPair(int first, int second) {
			set(first, second);
		}
		
		public void set(int left, int right) {
			first = left;
			second = right;
		}
		
		public int getFirst(){
			return first;
		}
		
		public int getSecond() {
			return second;
		}
		
		@Override
		public void readFields(DataInput in) throws IOException {
			first = in.readInt();
			second = in.readInt();
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(first);
			out.writeInt(second	);
		}
		
		@Override
		public int hashCode() {
			String p = Integer.toString(first)+ " " + Integer.toString(second);
			return p.hashCode();
		}
		
		@Override
		public int compareTo(IntPair o) {
			int thisfirst = first;
			int thissecond = second;
			int thatfirst = o.getFirst();
			int thatsecond = o.getSecond();
			if (thisfirst < thatfirst)
				return -1;
			else if (thisfirst > thatfirst)
				return 1;
			else {
				if (thissecond < thatsecond)
					return -1;
				else if (thissecond > thatsecond)
					return 1;
				else
					return 0;
			}
		}
    }
	
    // First Mapper for finding “similar” id pairs
	public static class First_Mapper extends Mapper<Object, Text, IntWritable, Text> {
		private IntWritable m_key = new IntWritable();
		private Text m_value = new Text();
		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // get threshold
            double TH = Double.parseDouble(context.getConfiguration().get("TH"));
            // get id and record
			String[] line = value.toString().split("\\s+");
			int k = 0;
			if (line.length > 1) {
                // compute prefix length
				int prefix = line.length - (int) Math.ceil((line.length - 1)*TH);
                // in case the prefix length is longer than total length
				if (prefix == line.length)
					--prefix;
                // get first n tokens, n is prefix length
				for (int i = 1; i <= prefix; ++i) {
                    // set key as token, value as total record
					k = Integer.parseInt(line[i]);
					m_key.set(k);
					m_value.set(value.toString());
					context.write(m_key, m_value);
				}
			}
		}
	}
	
    // First Patitioner for finding “similar” id pairs
	public static class First_Patitioner extends Partitioner<IntWritable, Text>{
		@Override
		public int getPartition(IntWritable arg0, Text arg1, int arg2) {
			// return an int less than number of reducer based on hash code
			return (arg0.hashCode() & Integer.MAX_VALUE) % arg2;
		}
	}
	
    // First Reducer for finding “similar” id pairs
	public static class First_Reducer extends Reducer<IntWritable, Text, Text, DoubleWritable> {
		private Text p = new Text();
		private DoubleWritable v = new DoubleWritable();
        @Override
		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // get threshold
            double TH = Double.parseDouble(context.getConfiguration().get("TH"));
            // store the previous record
			ArrayList<int[]> record_list = new ArrayList<int[]>();
            // pi store index of previous record, ci store index of current record
			int pi;
			int ci;
            // i_set store size of intersection set, u_set store size of union set
			double i_set;
			double u_set;
            // sim store similarity of two records
			double sim;
			for (Text value : values) {
                // get id and each record number
				String[] temp = value.toString().split("\\s+");
                // store these number in integer array
				int[] cur_r = new int[temp.length];
				for (int i = 0; i < temp.length; ++i) {
					cur_r[i] = Integer.parseInt(temp[i]);
				}
                // compare current record with all previous records in 'record_list'
				for (int[] pre_r : record_list) {
                    // initialization
					pi = 1;
					ci = 1;
					i_set = 0;
					u_set = 0;
                    /* calculate i_set and u_set of two records
                     * since the records list are ordered from small to large
                     * in worst case, go through each list once is enough for calculate i_set and u_set
                     * start with first number of each record, get out of loop if any record reach end
                     */
					while (pi < pre_r.length && ci < cur_r.length) {
                        // if their numbers are equal, both move to next number
                        // union set and intersection set sizes both plus 1
						if (pre_r[pi] == cur_r[ci]) {
							++u_set;
							++i_set;
							++pi;
							++ci;
						} else {
                            // if not equal, only union set size plus 1
                            // the smaller number's reocrd move to next number
							++u_set;
							if (pre_r[pi] < cur_r[ci])
								++pi;
							else
								++ci;
						}
					}
                    // if record list still have elements, union set size plus the number of elements
					if (pi < pre_r.length) {
						u_set += pre_r.length - pi;
					}
					if (ci < cur_r.length) {
						u_set += cur_r.length - ci;
					}
                    // calculate the similarity and compare to threshold
					sim = i_set / u_set;
					if (sim >= TH) {
                        // set key with ids, the smaller id places at first
						if (pre_r[0] < cur_r[0])
							p.set(Integer.toString(pre_r[0]) + "-" + Integer.toString(cur_r[0]));
						else
							p.set(Integer.toString(cur_r[0]) + "-" + Integer.toString(pre_r[0]));
                        // set value as similarity
						v.set(sim);
						context.write(p, v);
					}
				}
                // add current record to 'record list' (now it becomes one of the previous records)
				record_list.add(cur_r);
			}
		}
	}
	
    // Second Mapper for removing the duplicates
	public static class Second_Mapper extends Mapper<Object, Text, IntPair, Text> {
		private IntPair m_key = new IntPair();
		private Text m_value = new Text();
		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // get id pair and similarity
			String[] line = value.toString().split("\\s+");
            // get each id and set them to IntPair as key
			String[] pair = line[0].split("-");
			m_key.set(Integer.parseInt(pair[0]), Integer.parseInt(pair[1]));
            // set value as similarity
			m_value.set(line[1]);
			context.write(m_key, m_value);
		}
	}
	
    // Second Partitioner for removing the duplicates
	public static class Second_Patitioner extends Partitioner<IntPair, Text>{
		@Override
		public int getPartition(IntPair arg0, Text arg1, int arg2) {
			// return an int less than number of reducer based on hash code
			return (arg0.hashCode() & Integer.MAX_VALUE) % arg2;
		}
	}
	
    // Second Reducer for removing the duplicates
	public static class Second_Reducer extends Reducer<IntPair, Text, Text, Text> {
		private Text r_key = new Text();
        @Override
		public void reduce(IntPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // only process with first value (all the rest are duplicates)
			for (Text value : values) {
                // set id pair as required format
				r_key.set("(" + Integer.toString(key.getFirst()) + "," + Integer.toString(key.getSecond()) + ")");
				context.write(r_key, value);
				break;
			}
		}
	}
	
	public static void main(String[] args) throws Exception { 
		// get 3 parameters and set temp output
		String IN = args[0];
		String OUT = args[1];
		String temp = OUT + "1";
		int RDN = Integer.parseInt(args[3]);
		
        // First MapReduce for finding “similar” id pairs
        // the output is temp
		Configuration conf_pre = new Configuration();
		conf_pre.set("TH", args[2]);
   		Job job_pre = Job.getInstance(conf_pre, "Pre-Operation");
   		job_pre.setJarByClass(SetSimJoin.class);
   		job_pre.setMapperClass(First_Mapper.class);
   		job_pre.setPartitionerClass(First_Patitioner.class);
   		job_pre.setReducerClass(First_Reducer.class);
   		job_pre.setMapOutputKeyClass(IntWritable.class);
   		job_pre.setMapOutputValueClass(Text.class);
   		job_pre.setOutputKeyClass(Text.class);
   		job_pre.setOutputValueClass(DoubleWritable.class);
   		job_pre.setNumReduceTasks(RDN);
   		FileInputFormat.addInputPath(job_pre, new Path(IN));
   		FileOutputFormat.setOutputPath(job_pre, new Path(temp));
   		job_pre.waitForCompletion(true);
   		
        // Second MapReduce for removing duplicates
   		Configuration conf_last = new Configuration();
   		Job job_last = Job.getInstance(conf_last, "Last-Operation");
   		job_last.setJarByClass(SetSimJoin.class);
   		job_last.setMapperClass(Second_Mapper.class);
   		job_last.setPartitionerClass(Second_Patitioner.class);
   		job_last.setReducerClass(Second_Reducer.class);
   		job_last.setMapOutputKeyClass(IntPair.class);
   		job_last.setMapOutputValueClass(Text.class);
   		job_last.setOutputKeyClass(Text.class);
   		job_last.setOutputValueClass(Text.class);
   		job_last.setNumReduceTasks(RDN);
   		FileInputFormat.addInputPath(job_last, new Path(temp));
   		FileOutputFormat.setOutputPath(job_last, new Path(OUT));
   		job_last.waitForCompletion(true);
	}
}
